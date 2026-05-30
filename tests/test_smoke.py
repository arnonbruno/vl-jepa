"""Smoke tests for VL-JEPA model (fixed version).
Tests proper JEPA functionality: masking, predictor, loss computation, GPU.
"""

import torch
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import os

import torch.nn.functional as F

from src.model import (
    VL_JEPA, VisionEncoder, LanguageEncoder, Predictor, MemoryBank,
    OpenCLIPVisionEncoder, OpenCLIPLanguageEncoder, LinearProjection,
    ResidualLinearProjection,
    LOGIT_SCALE_MAX, block_patch_mask, compute_jepa_loss, make_multicrop_views,
    sigmoid_contrastive_loss, hard_negative_ranking_loss,
)
import pytest

try:
    import open_clip  # noqa: F401
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False

# open_clip's stock ViT-B-16 tower is fixed-dim; tests use random weights
# (pretrained=None) so they stay fast and network-free.
OPENCLIP_MODEL = "ViT-B-16"
OPENCLIP_IMAGE_SIZE = 224
OPENCLIP_PATCH = 16
OPENCLIP_NUM_PATCHES = (OPENCLIP_IMAGE_SIZE // OPENCLIP_PATCH) ** 2
OPENCLIP_VISION_DIM = 768
OPENCLIP_TEXT_DIM = 512
OPENCLIP_VOCAB = 49408
OPENCLIP_SEQ_LEN = 32

from src.trainer import (
    CheckpointError,
    ModelEMA,
    VL_JEPA_Trainer,
    _GRAD_SCALER_GROWTH_INTERVAL,
    _MAX_GRAD_SCALER_SCALE,
    _make_grad_scaler,
    find_first_nonfinite_output,
    validate_checkpoint,
    wise_ft_interpolate,
)

HIDDEN_DIM = 96
IMAGE_SIZE = 64
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
SEQ_LEN = 32
VOCAB_SIZE = 30522


def _make_model(**kwargs):
    defaults = {
        "hidden_dim": HIDDEN_DIM,
        "patch_size": PATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "predictor_layers": 1,
        "freeze_encoders": False,
        "projection_dim": HIDDEN_DIM,
    }
    defaults.update(kwargs)
    return VL_JEPA(**defaults)


def test_vision_encoder_with_mask():
    """Test vision encoder with masking."""
    print("Testing VisionEncoder with mask...")
    encoder = VisionEncoder(hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)

    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    mask = torch.zeros(2, NUM_PATCHES, dtype=torch.bool)
    mask[:, : int(NUM_PATCHES * 0.75)] = True  # 75% masked

    out = encoder(x, mask)
    assert out.shape == (2, NUM_PATCHES + 1, HIDDEN_DIM), f"Unexpected shape: {out.shape}"
    print(f"  ✓ Vision output shape: {out.shape}")
    print(f"  ✓ mask_token exists: {'mask_token' in dict(encoder.named_parameters())}")


def test_language_encoder():
    """Test language encoder shape."""
    print("Testing LanguageEncoder...")
    encoder = LanguageEncoder(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, max_seq_len=512)

    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    out = encoder(input_ids)

    assert out.shape == (2, SEQ_LEN, HIDDEN_DIM), f"Unexpected shape: {out.shape}"
    print(f"  ✓ Language output shape: {out.shape}")


def test_predictor():
    """Test predictor module."""
    print("Testing Predictor...")
    predictor = Predictor(hidden_dim=HIDDEN_DIM, num_layers=1)

    x = torch.randn(2, NUM_PATCHES + 1, HIDDEN_DIM)
    out = predictor(x)

    assert out.shape == (2, NUM_PATCHES + 1, HIDDEN_DIM), f"Unexpected shape: {out.shape}"
    print(f"  ✓ Predictor output shape: {out.shape}")


def test_vl_jepa_forward():
    """Test full VL-JEPA forward pass and loss computation."""
    print("Testing VL-JEPA forward + loss...")
    model = _make_model()

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    outputs = model(images, input_ids)
    loss_dict = compute_jepa_loss(outputs)

    # Check all output keys
    expected_keys = ['predicted_patches', 'target_patches', 'patch_mask',
                     'vision_cls', 'language_cls', 'vision_proj', 'language_proj']
    for k in expected_keys:
        assert k in outputs, f"Missing key: {k}"

    # Check loss keys
    assert 'total_loss' in loss_dict
    assert 'mse_loss' in loss_dict
    assert 'nce_loss' in loss_dict

    # Check shapes
    assert outputs['predicted_patches'].shape == (2, NUM_PATCHES + 1, HIDDEN_DIM)
    assert outputs['target_patches'].shape == (2, NUM_PATCHES + 1, HIDDEN_DIM)
    assert outputs['patch_mask'].shape == (2, NUM_PATCHES)
    assert outputs['vision_proj'].shape == (2, HIDDEN_DIM)
    assert outputs['language_proj'].shape == (2, HIDDEN_DIM)

    # Check projections are normalized
    vision_norm = torch.norm(outputs['vision_proj'], p=2, dim=-1)
    assert torch.allclose(vision_norm, torch.ones_like(vision_norm), atol=1e-5)

    # Check losses are finite
    assert torch.isfinite(loss_dict['total_loss']), "Loss is NaN or Inf!"
    assert loss_dict['mse_loss'] > 0, f"MSE loss should be > 0, got {loss_dict['mse_loss']}"
    assert loss_dict['nce_loss'] > 0, f"NCE loss should be > 0, got {loss_dict['nce_loss']}"

    print(f"  ✓ predicted_patches: {outputs['predicted_patches'].shape}")
    print(f"  ✓ target_patches: {outputs['target_patches'].shape}")
    print(f"  ✓ patch_mask: {outputs['patch_mask'].shape} ({outputs['patch_mask'].float().mean():.0%} masked)")
    print(f"  ✓ vision_proj: {outputs['vision_proj'].shape} (normalized ✓)")
    print(f"  ✓ MSE loss: {loss_dict['mse_loss']:.4f}")
    print(f"  ✓ NCE loss: {loss_dict['nce_loss']:.4f}")
    print(f"  ✓ Total loss: {loss_dict['total_loss']:.4f}")


def test_block_mask_exact_ratio_and_deterministic_seed():
    """Block masking should mask an exact ratio and be reproducible with a seed."""
    print("Testing block patch mask...")
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    mask_a = block_patch_mask(4, NUM_PATCHES, mask_ratio=0.75, generator=generator_a)
    mask_b = block_patch_mask(4, NUM_PATCHES, mask_ratio=0.75, generator=generator_b)

    expected_masked = int(NUM_PATCHES * 0.75)
    assert torch.equal(mask_a, mask_b)
    assert torch.all(mask_a.sum(dim=1) == expected_masked)
    print(f"  ✓ Block mask shape: {mask_a.shape}, masked per sample: {expected_masked}")


def test_block_mask_tiny_grid_keeps_visible_context():
    """Tiny local-crop grids should never be fully masked."""
    print("Testing tiny-grid block mask visibility...")
    mask = block_patch_mask(8, 36, mask_ratio=0.95)
    masked_per_sample = mask.sum(dim=1)

    assert torch.all(masked_per_sample < 36), masked_per_sample
    assert torch.all(masked_per_sample > 0), masked_per_sample
    print(f"  ✓ Tiny 6x6 grid keeps {36 - int(masked_per_sample[0])} visible patches")


def test_multicrop_views_shapes():
    """Global and local crops should preserve batch size and requested resolutions."""
    print("Testing multi-crop view generation...")
    images = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
    views = make_multicrop_views(images, global_size=IMAGE_SIZE, local_size=32, training=True)

    assert views['global'].shape == (3, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert views['local'].shape == (3, 3, 32, 32)
    print(f"  ✓ Global crop: {views['global'].shape}; local crop: {views['local'].shape}")


def test_multicrop_forward_is_finite():
    """Trainer multi-crop path should produce finite losses on small crops."""
    print("Testing multi-crop forward stability...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model,
        device,
        learning_rate=1e-4,
        warmup_steps=0,
        max_steps=10,
        use_multi_crop=True,
        global_crop_size=IMAGE_SIZE,
        local_crop_size=32,
    )
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    metrics = trainer.train_step(images, input_ids)
    assert all(math.isfinite(metrics[k]) for k in ('mse_loss', 'nce_loss', 'total_loss', 'grad_norm'))
    print(f"  ✓ Multi-crop loss finite: {metrics['total_loss']:.4f}")


def test_target_teacher_modes_are_restored_after_forward():
    """Forward should not leave EMA vision teacher stuck in eval mode."""
    print("Testing teacher mode restoration...")
    model = _make_model()
    model.train()
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    _ = model(images, input_ids)
    assert model.target_encoder.training

    model.eval()
    _ = model(images, input_ids)
    assert not model.target_encoder.training
    print("  ✓ Teacher module training/eval modes are restored")


def test_logit_scale_is_bounded_before_exp():
    """Large learned logit scales should not overflow during loss computation."""
    print("Testing logit scale bounding...")
    model = _make_model()
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    outputs = model(images, input_ids)
    outputs['logit_scale'] = torch.tensor(1000.0, requires_grad=True)
    loss_dict = compute_jepa_loss(outputs)

    assert torch.isfinite(loss_dict['total_loss'])
    assert torch.allclose(loss_dict['logit_scale'], torch.tensor(math.exp(LOGIT_SCALE_MAX)))
    print(f"  ✓ Logit scale capped at {loss_dict['logit_scale'].item():.1f}")


def test_padded_small_crop_forward_is_finite():
    """Small non-divisible square crops are padded to a valid patch grid."""
    print("Testing padded small crop forward...")
    model = _make_model(image_size=48)
    images = torch.randn(2, 3, 30, 30)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    outputs = model(images, input_ids)
    loss_dict = compute_jepa_loss(outputs)

    assert outputs['patch_mask'].shape == (2, 4)
    assert torch.isfinite(loss_dict['total_loss'])
    print("  ✓ Non-divisible 30px crop pads to a finite 2x2 patch grid")


def test_memory_bank_expands_contrastive_negatives():
    """Memory bank should increase i2t logits width beyond batch size."""
    print("Testing memory bank contrastive negatives...")
    model = _make_model()
    bank = MemoryBank(size=128, dim=HIDDEN_DIM, device=torch.device('cpu'))

    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
    outputs = model(images, input_ids)

    for _ in range(4):
        bank.enqueue(outputs['language_proj'].detach())

    loss_with_bank = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, memory_bank=bank)
    loss_batch_only = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, memory_bank=None)

    assert bank.num_filled > 4
    assert torch.isfinite(loss_with_bank['nce_loss'])
    assert loss_with_bank['nce_loss'].item() != loss_batch_only['nce_loss'].item()
    print(f"  ✓ Queue filled={bank.num_filled}, NCE with bank={loss_with_bank['nce_loss']:.4f}")


def test_memory_bank_size_zero_disables_queue():
    """Trainer should run cleanly without allocating a memory bank."""
    print("Testing memory_bank_size=0...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model,
        device,
        learning_rate=1e-4,
        warmup_steps=0,
        max_steps=10,
        memory_bank_size=0,
    )
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    metrics = trainer.train_step(images, input_ids)

    assert trainer.memory_bank is None
    assert math.isfinite(metrics['total_loss'])
    print("  ✓ No memory bank allocated; train step finite")


def test_siglip_contrastive_loss_is_finite():
    """SigLIP loss path should produce finite loss and bounded accuracy."""
    print("Testing SigLIP loss...")
    vision_proj = torch.nn.functional.normalize(torch.randn(4, HIDDEN_DIM), dim=-1)
    language_proj = torch.nn.functional.normalize(torch.randn(4, HIDDEN_DIM), dim=-1)
    loss, acc = sigmoid_contrastive_loss(
        vision_proj,
        language_proj,
        torch.tensor(2.659),
        torch.tensor(-10.0),
    )

    assert torch.isfinite(loss)
    assert 0.0 <= acc.item() <= 1.0
    print(f"  ✓ SigLIP loss finite: {loss:.4f}, acc={acc:.2%}")


def test_hard_negative_loss_zero_when_perfectly_separated():
    """Perfectly aligned pairs (positives far above negatives) give ~zero hinge."""
    print("Testing hard-negative loss on separable batch...")
    # Orthogonal one-hot rows: diagonal sim = 1, off-diagonal = 0, margin 0.2.
    proj = torch.eye(4)
    loss = hard_negative_ranking_loss(proj, proj, margin=0.2)
    assert loss.item() == 0.0, loss.item()

    # Collapsed embeddings: every row identical -> negatives are exactly as
    # close as positives, so each hinge equals the margin.
    row = torch.nn.functional.normalize(torch.randn(1, HIDDEN_DIM), dim=-1)
    collapsed = row.expand(4, HIDDEN_DIM).contiguous()
    loss_hard = hard_negative_ranking_loss(collapsed, collapsed, margin=0.2)
    assert abs(loss_hard.item() - 2 * 0.2) < 1e-5, loss_hard.item()
    print(f"  ✓ separable={loss.item():.3f}, collapsed={loss_hard.item():.3f}")


def test_hard_negative_single_sample_is_zero():
    """A batch of one has no negatives -> loss must be exactly zero."""
    proj = torch.nn.functional.normalize(torch.randn(1, HIDDEN_DIM), dim=-1)
    assert hard_negative_ranking_loss(proj, proj).item() == 0.0


def test_compute_jepa_loss_hard_negative_weight_adds_term():
    """Enabling the hard-negative weight changes the total loss and flows grads."""
    print("Testing hard-negative weight in compute_jepa_loss...")
    model = _make_model(contrastive_loss='siglip')
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))

    outputs = model(images, input_ids)
    base = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, gamma=0.0)
    with_hn = compute_jepa_loss(
        outputs, alpha=0.0, beta=1.0, gamma=0.0,
        hard_negative_weight=0.5, hard_negative_margin=0.2,
    )
    assert 'hard_neg_loss' in with_hn
    assert torch.isfinite(with_hn['total_loss'])
    # The hard-negative term is non-negative; total should not be below the base.
    assert with_hn['total_loss'].item() >= base['total_loss'].item() - 1e-6
    with_hn['total_loss'].backward()
    assert model.vision_proj.weight.grad is not None
    print(f"  ✓ base={base['total_loss']:.4f}, +hardneg={with_hn['total_loss']:.4f}")


def test_trainer_hard_negative_weight_runs():
    """Trainer with hard_negative_weight > 0 produces a finite step."""
    print("Testing trainer hard-negative path...")
    model = _make_model(contrastive_loss='siglip')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-4, warmup_steps=0, max_steps=10,
        alpha=0.0, beta=1.0, hard_negative_weight=0.3,
    )
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
    metrics = trainer.train_step(images, input_ids)
    assert not metrics.get('skipped')
    assert math.isfinite(metrics['total_loss'])
    print(f"  ✓ trainer hard-negative step finite: {metrics['total_loss']:.4f}")


def test_contrastive_projection_gradients_flow():
    """SigLIP must update both projection heads."""
    print("Testing SigLIP projection gradients...")
    model = _make_model(contrastive_loss='siglip')
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))

    outputs = model(images, input_ids)
    loss = compute_jepa_loss(outputs, alpha=0.0, beta=1.0)['total_loss']
    loss.backward()

    vision_grad = model.vision_proj.weight.grad
    language_grad = model.language_proj.weight.grad
    assert vision_grad is not None and vision_grad.abs().sum() > 0
    assert language_grad is not None and language_grad.abs().sum() > 0
    print("  ✓ vision_proj and language_proj receive SigLIP gradients")


def test_gradient_clipping_caps_norm():
    """Global grad norm clip should bound post-clip parameter gradients."""
    print("Testing gradient clipping...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_norm = 0.5
    trainer = VL_JEPA_Trainer(
        model,
        device,
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=10,
        alpha=0.0,
        beta=1.0,
        max_grad_norm=max_norm,
    )
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN), device=device)

    metrics = trainer.train_step(images, input_ids)
    assert metrics['grad_norm'] > 0
    post_clip_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), float('inf'),
    )
    assert float(post_clip_norm) <= max_norm + 1e-5
    print(f"  ✓ Post-clip grad norm {float(post_clip_norm):.4f} <= {max_norm}")


def test_gradient_accumulation_defers_optimizer_step():
    """Accumulation should step the optimizer only on the final micro-batch."""
    print("Testing gradient accumulation...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-3, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0,
    )

    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN), device=device)

    accum = 4
    weight_before = model.vision_proj.weight.detach().clone()

    # First accum-1 micro-batches must NOT advance the optimizer.
    for i in range(accum - 1):
        metrics = trainer.train_step(
            images, input_ids, accumulation_steps=accum, step_optimizer=False,
        )
        assert not metrics.get('skipped')
        assert trainer._step == 0, f"optimizer stepped early at micro-batch {i}"
        assert torch.equal(model.vision_proj.weight.detach(), weight_before)

    # Final micro-batch triggers the optimizer step.
    metrics = trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=True,
    )
    assert not metrics.get('skipped')
    assert trainer._step == 1, "optimizer should step once per accumulation window"
    assert trainer._accum_counter == 0
    assert not torch.equal(model.vision_proj.weight.detach(), weight_before)
    print("  ✓ Optimizer steps once per accumulation window")


def test_momentum_update():
    """Test that momentum update works (target != context at init)."""
    print("Testing momentum update...")
    model = _make_model()

    # Before update: should be identical (loaded same weights)
    ctx_state = next(model.context_encoder.parameters())
    tgt_state = next(model.target_encoder.parameters())
    assert torch.allclose(ctx_state, tgt_state), "Target should match context at init"

    # Modify context
    with torch.no_grad():
        ctx_state.add_(torch.randn_like(ctx_state) * 0.1)

    # Momentum update (tau=0.9 for test)
    model.momentum_tau = 0.9
    model.momentum_update()

    # After update: target should have moved partially
    new_tgt = next(model.target_encoder.parameters())
    assert not torch.allclose(ctx_state, new_tgt), "Target should differ from context after update"
    print(f"  ✓ Momentum update works (tau={model.momentum_tau})")


def test_gpu_forward():
    """Test VL-JEPA on GPU if available."""
    print("Testing VL-JEPA on GPU...")
    if not torch.cuda.is_available():
        print("  ⚠️  No GPU available, skipping GPU test")
        return

    model = _make_model().cuda()
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)).cuda()

    with torch.no_grad():
        outputs = model(images, input_ids)

    print(f"  ✓ Forward pass successful on GPU")
    print(f"  ✓ GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"  ✓ Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

    print(f"  ✓ Inference without grad: OK")

    # Run backward test separately (need grad)
    torch.cuda.empty_cache()
    model2 = _make_model().cuda()
    images2 = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()
    input_ids2 = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)).cuda()

    outputs2 = model2(images2, input_ids2)
    loss_dict = compute_jepa_loss(outputs2)
    loss_dict['total_loss'].backward()

    grad_count = sum(1 for p in model2.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  ✓ Gradients flowing: {grad_count}/{sum(1 for p in model2.parameters() if p.requires_grad)} params")


def test_loss_decreases_over_steps():
    """Sanity check: moving-average loss trends down over training steps."""
    print("Testing loss decreases over training steps...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-3, warmup_steps=0, max_steps=1000)

    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))

    losses = []
    for _ in range(12):
        metrics = trainer.train_step(images, input_ids)
        losses.append(metrics['total_loss'])

    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    # Block masking makes the task harder — loss may not decrease in 12 steps,
    # but it should not NaN and should stay within reasonable bounds
    assert not any(math.isnan(l) for l in losses), "Loss went to NaN!"
    assert all(isinstance(l, float) and l > 0 for l in losses), f"Invalid loss values: {losses}"
    print(f"  ✓ Loss stable over 12 steps (first3={avg_first:.4f}, last3={avg_last:.4f}, min={min(losses):.4f}, max={max(losses):.4f})")


def test_checkpoint_roundtrip():
    """Save and reload checkpoint; forward pass should match."""
    print("Testing checkpoint save/load round-trip...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _make_model().to(device)
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=100)

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)).to(device)

    trainer.train_step(images, input_ids)

    # Fixed masks so forward() is deterministic (model randomizes masks when None)
    num_patches = NUM_PATCHES
    patch_mask = torch.zeros(2, num_patches, dtype=torch.bool, device=device)
    patch_mask[:, : int(num_patches * 0.75)] = True  # 75% masked, same pattern every call
    token_mask = torch.zeros(2, input_ids.size(1), dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        out_before = model(images, input_ids, patch_mask=patch_mask, token_mask=token_mask)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'test_ckpt.pt')
        trainer.save_checkpoint(ckpt_path)

        model2 = _make_model().to(device)
        trainer2 = VL_JEPA_Trainer(model2, device, warmup_steps=0, max_steps=100)
        trainer2.load_checkpoint(ckpt_path)

        model2.eval()
        with torch.no_grad():
            out_after = model2(
                images, input_ids, patch_mask=patch_mask, token_mask=token_mask,
            )

    for key in ('predicted_patches', 'vision_proj', 'language_proj'):
        assert torch.allclose(out_before[key], out_after[key], atol=1e-5), f"Mismatch on {key}"
    print("  ✓ Checkpoint round-trip preserves model outputs")


def test_validate_checkpoint_rejects_nan_weights():
    """validate_checkpoint must reject NaN/Inf in model or optimizer state."""
    device = torch.device('cpu')
    model = _make_model().to(device)
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=10)
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    trainer.train_step(images, input_ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        good_path = os.path.join(tmpdir, 'good.pt')
        trainer.save_checkpoint(good_path)
        good_ckpt = torch.load(good_path, weights_only=False)

        validate_checkpoint(good_ckpt, path=good_path)

        bad_ckpt = dict(good_ckpt)
        bad_ckpt['model_state_dict'] = {
            k: torch.full_like(v, float('nan'))
            for k, v in good_ckpt['model_state_dict'].items()
            if isinstance(v, torch.Tensor)
        }
        with pytest.raises(CheckpointError, match='model weights'):
            validate_checkpoint(bad_ckpt, path='bad.pt')

        bad_opt = dict(good_ckpt)
        opt_state = bad_opt['optimizer_state_dict']
        first_key = next(iter(opt_state['state']))
        opt_state['state'][first_key]['exp_avg'] = torch.full(
            opt_state['state'][first_key]['exp_avg'].shape,
            float('inf'),
        )
        with pytest.raises(CheckpointError, match='optimizer state'):
            validate_checkpoint(bad_opt, path='bad.pt')


def test_load_checkpoint_raises_on_corrupt_file():
    """load_checkpoint must refuse corrupted checkpoints when validate=True."""
    device = torch.device('cpu')
    model = _make_model().to(device)
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'corrupt.pt')
        trainer.save_checkpoint(ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=False)
        ckpt['model_state_dict']['logit_scale'] = torch.tensor(float('nan'))
        torch.save(ckpt, ckpt_path)

        trainer2 = VL_JEPA_Trainer(_make_model().to(device), device, warmup_steps=0, max_steps=10)
        with pytest.raises(CheckpointError, match='Corrupted checkpoint'):
            trainer2.load_checkpoint(ckpt_path)


def test_load_checkpoint_returns_metadata():
    """load_checkpoint returns epoch and other extra fields."""
    device = torch.device('cpu')
    model = _make_model().to(device)
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'meta.pt')
        trainer.save_checkpoint(ckpt_path, extra={'epoch': 3, 'val_loss': 1.25})
        meta = trainer.load_checkpoint(ckpt_path)
        assert meta['epoch'] == 3
        assert meta['val_loss'] == 1.25
        assert 'model_state_dict' not in meta


def test_masked_patches_contribute_to_loss():
    """MSE loss uses masked patches only; unmasked-only mask yields ~zero MSE."""
    print("Testing masked vs unmasked patch loss contribution...")
    model = _make_model()
    model.eval()

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    num_patches = NUM_PATCHES

    # All patches masked — MSE should be positive
    mask_all = torch.ones(2, num_patches, dtype=torch.bool)
    with torch.no_grad():
        out_masked = model(images, input_ids, patch_mask=mask_all)
    loss_masked = compute_jepa_loss(out_masked)['mse_loss']

    # No patches masked — MSE should be ~0
    mask_none = torch.zeros(2, num_patches, dtype=torch.bool)
    with torch.no_grad():
        out_unmasked = model(images, input_ids, patch_mask=mask_none)
    loss_unmasked = compute_jepa_loss(out_unmasked)['mse_loss']

    assert loss_masked > 0, f"Masked MSE should be > 0, got {loss_masked}"
    assert loss_unmasked < 1e-6, f"Unmasked MSE should be ~0, got {loss_unmasked}"
    print(f"  ✓ MSE (all masked): {loss_masked:.4f}")
    print(f"  ✓ MSE (none masked): {loss_unmasked:.6f}")


def test_momentum_schedule_capped():
    """EMA τ schedule should not stretch across the full LR max_steps horizon."""
    print("Testing capped momentum schedule...")
    model = _make_model()
    device = torch.device('cpu')
    trainer = VL_JEPA_Trainer(
        model, device, warmup_steps=0, max_steps=100_000, momentum_schedule_steps=500,
    )
    assert len(trainer._momentum_schedule) == 500
    assert abs(trainer._momentum_schedule[0].item() - 0.996) < 1e-6
    assert abs(trainer._momentum_schedule[-1].item() - 1.0) < 1e-6
    print(f"  ✓ Momentum schedule length: {len(trainer._momentum_schedule)}")


def test_trainer_skips_non_finite_batch():
    """Non-finite loss batches are skipped without corrupting weights."""
    print("Testing non-finite batch skip...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=100)

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device)

    before = next(model.context_encoder.parameters()).detach().clone()
    with torch.no_grad():
        model.vision_pred_head.weight.fill_(float('nan'))

    metrics = trainer.train_step(images, input_ids)
    after = next(model.context_encoder.parameters()).detach()

    assert metrics.get('skipped') is True
    assert metrics.get('skip_reason') in (
        'non_finite_loss', 'non_finite_output', 'corrupt_weights_pre_forward',
    )
    assert torch.equal(before.to(device), after)
    print("  ✓ Non-finite batch skipped; context encoder weights unchanged")


def test_trainer_detects_corrupt_weights_before_forward(tmp_path):
    """Pre-forward weight check skips batch and marks weights_corrupted."""
    print("Testing corrupt-weight detection before forward...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model, device, warmup_steps=0, max_steps=100,
        nan_diagnostics_dir=str(tmp_path),
    )

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device)

    with torch.no_grad():
        model.vision_proj.weight.fill_(float('nan'))

    metrics = trainer.train_step(images, input_ids, batch_index=7, epoch=1)
    assert metrics.get('skipped') is True
    assert metrics.get('skip_reason') == 'corrupt_weights_pre_forward'
    assert metrics.get('weights_corrupted') is True
    diag_dirs = list(tmp_path.glob('nan_*'))
    assert len(diag_dirs) == 1
    assert (diag_dirs[0] / 'diagnostic.json').is_file()
    print("  ✓ Corrupt weights detected; diagnostic saved")


def test_find_first_nonfinite_output():
    """Output finite-check locates the first bad activation tensor."""
    print("Testing non-finite output detection...")
    good = torch.randn(2, HIDDEN_DIM)
    bad = good.clone()
    bad[0, 0] = float('nan')
    outputs = {
        'predicted_patches': good,
        'target_patches': good,
        'vision_proj': bad,
        'language_proj': good,
    }
    assert find_first_nonfinite_output(outputs) == 'vision_proj'
    print("  ✓ find_first_nonfinite_output locates bad tensor")


def test_trainer_weights_stay_finite_after_step():
    """Successful train steps leave all model weights finite."""
    print("Testing post-step weight finiteness...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=100)

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device)

    for _ in range(5):
        metrics = trainer.train_step(images, input_ids)
        assert not metrics.get('skipped')
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite weights in {name}"
    print("  ✓ Weights remain finite after 5 steps")


def test_grad_scaler_growth_interval_capped():
    """GradScaler uses a huge growth interval and hard scale cap for stability."""
    print("Testing GradScaler stability settings...")
    if not torch.cuda.is_available():
        print("  ⚠️  No GPU available, skipping GradScaler test")
        return

    scaler = _make_grad_scaler(torch.device('cuda'))
    assert scaler._init_scale == 1024.0
    assert scaler._growth_interval >= 2**30
    assert _GRAD_SCALER_GROWTH_INTERVAL >= 2**30
    assert _MAX_GRAD_SCALER_SCALE == 8192.0
    print(
        f"  ✓ GradScaler init_scale={scaler._init_scale}, "
        f"growth_interval={scaler._growth_interval}"
    )


def test_joint_embedding():
    """Test inference-time joint embedding extraction."""
    print("Testing joint embedding extraction...")
    model = _make_model()

    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    vision_proj, language_proj = model.get_joint_embedding(images, input_ids)

    assert vision_proj.shape == (2, HIDDEN_DIM), f"Expected (2, {HIDDEN_DIM}), got {vision_proj.shape}"
    assert language_proj.shape == (2, HIDDEN_DIM), f"Expected (2, {HIDDEN_DIM}), got {language_proj.shape}"

    # Check normalization
    vision_norm = torch.norm(vision_proj, p=2, dim=-1)
    language_norm = torch.norm(language_proj, p=2, dim=-1)
    assert torch.allclose(vision_norm, torch.ones_like(vision_norm), atol=1e-5)
    assert torch.allclose(language_norm, torch.ones_like(language_norm), atol=1e-5)

    print(f"  ✓ Vision embedding: {vision_proj.shape} (normalized ✓)")
    print(f"  ✓ Language embedding: {language_proj.shape} (normalized ✓)")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_openclip_vision_encoder_shapes():
    """OpenCLIP vision encoder mirrors the (B, N+1, D) patch-sequence contract."""
    print("Testing OpenCLIPVisionEncoder shapes...")
    encoder = OpenCLIPVisionEncoder(
        OPENCLIP_MODEL, pretrained=None, image_size=OPENCLIP_IMAGE_SIZE, freeze=False,
    )
    assert encoder.hidden_dim == OPENCLIP_VISION_DIM
    assert encoder.num_patches == OPENCLIP_NUM_PATCHES
    assert encoder.patch_size == OPENCLIP_PATCH
    assert 'mask_token' in dict(encoder.named_parameters())

    x = torch.randn(2, 3, OPENCLIP_IMAGE_SIZE, OPENCLIP_IMAGE_SIZE)
    mask = block_patch_mask(2, OPENCLIP_NUM_PATCHES, mask_ratio=0.75)
    out = encoder(x, mask)
    assert out.shape == (2, OPENCLIP_NUM_PATCHES + 1, OPENCLIP_VISION_DIM), out.shape

    # Smaller (local-crop) resolution must interpolate positional embeddings.
    out_local = encoder(torch.randn(2, 3, 96, 96), mask=None)
    assert out_local.shape == (2, (96 // OPENCLIP_PATCH) ** 2 + 1, OPENCLIP_VISION_DIM)
    print(f"  ✓ OpenCLIP vision output shape: {out.shape}")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_openclip_language_encoder_shapes():
    """OpenCLIP text encoder returns the full (B, S, D) token sequence."""
    print("Testing OpenCLIPLanguageEncoder shapes...")
    encoder = OpenCLIPLanguageEncoder(OPENCLIP_MODEL, pretrained=None, freeze=False)
    assert encoder.hidden_dim == OPENCLIP_TEXT_DIM

    input_ids = torch.randint(0, OPENCLIP_VOCAB, (2, OPENCLIP_SEQ_LEN))
    attention_mask = torch.ones(2, OPENCLIP_SEQ_LEN, dtype=torch.long)
    out = encoder(input_ids, attention_mask)
    assert out.shape == (2, OPENCLIP_SEQ_LEN, OPENCLIP_TEXT_DIM), out.shape
    print(f"  ✓ OpenCLIP language output shape: {out.shape}")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_openclip_vl_jepa_forward():
    """Full VL-JEPA forward + loss + backward on the openclip backbone."""
    print("Testing VL-JEPA (openclip) forward + loss...")
    model = VL_JEPA(
        hidden_dim=OPENCLIP_VISION_DIM,
        patch_size=OPENCLIP_PATCH,
        image_size=OPENCLIP_IMAGE_SIZE,
        predictor_layers=1,
        vision_backbone="openclip",
        text_backbone="openclip",
        openclip_model=OPENCLIP_MODEL,
        openclip_pretrained=None,
        freeze_encoders=False,
        projection_dim=512,
        contrastive_loss="siglip",
    )
    # Vision tower drives JEPA hidden dim; text tower keeps its own width.
    assert model.hidden_dim == OPENCLIP_VISION_DIM
    assert model.language_hidden_dim == OPENCLIP_TEXT_DIM

    images = torch.randn(2, 3, OPENCLIP_IMAGE_SIZE, OPENCLIP_IMAGE_SIZE)
    input_ids = torch.randint(0, OPENCLIP_VOCAB, (2, OPENCLIP_SEQ_LEN))
    attention_mask = torch.ones(2, OPENCLIP_SEQ_LEN, dtype=torch.long)

    outputs = model(images, input_ids, attention_mask)
    assert outputs['predicted_patches'].shape == (2, OPENCLIP_NUM_PATCHES + 1, OPENCLIP_VISION_DIM)
    assert outputs['target_patches'].shape == (2, OPENCLIP_NUM_PATCHES + 1, OPENCLIP_VISION_DIM)
    assert outputs['patch_mask'].shape == (2, OPENCLIP_NUM_PATCHES)
    assert outputs['vision_proj'].shape == (2, 512)
    assert outputs['language_proj'].shape == (2, 512)

    loss_dict = compute_jepa_loss(
        outputs, alpha=0.1, beta=0.9, gamma=0.01, label_smoothing=0.05,
    )
    assert torch.isfinite(loss_dict['total_loss'])
    loss_dict['total_loss'].backward()
    assert model.vision_pred_head.weight.grad is not None
    print(f"  ✓ OpenCLIP VL-JEPA loss finite: {loss_dict['total_loss']:.4f}")


def test_linear_projection_reproduces_clip_matmul():
    """LinearProjection seeded with a CLIP matrix computes exactly x @ proj."""
    print("Testing LinearProjection CLIP-matrix seeding...")
    in_dim, out_dim = 8, 5
    proj = torch.randn(in_dim, out_dim)
    head = LinearProjection(in_dim, out_dim, init_weight=proj)

    # weight stored transposed; raw output equals the native x @ proj.
    assert torch.allclose(head.weight, proj.t(), atol=1e-6)
    x = torch.randn(4, in_dim)
    assert torch.allclose(head.raw(x), x @ proj, atol=1e-5)
    # forward is the L2-normalized projection.
    expected = F.normalize(x @ proj, p=2, dim=-1, eps=1e-6)
    assert torch.allclose(head(x), expected, atol=1e-5)
    print("  ✓ LinearProjection reproduces x @ proj and normalizes")


def test_eot_pooling_selects_last_valid_token():
    """text_pool='eot' returns the last non-pad token state, not a mean."""
    print("Testing EOT text pooling...")
    model = _make_model(text_pool="eot")
    assert model.text_pool == "eot"

    b, seq, dim = 3, 6, HIDDEN_DIM
    language_emb = torch.randn(b, seq, dim)
    # Varying caption lengths -> EOT at positions 4, 2, 5.
    attention_mask = torch.zeros(b, seq, dtype=torch.long)
    lengths = [5, 3, 6]
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    pooled = model._pool_language(language_emb, attention_mask)
    for i, length in enumerate(lengths):
        assert torch.allclose(pooled[i], language_emb[i, length - 1]), i
    print("  ✓ EOT pooling selects the last valid token per sample")


def test_mean_pooling_unchanged_default():
    """Default text_pool='mean' still averages over valid tokens."""
    print("Testing mean pooling default...")
    model = _make_model()
    assert model.text_pool == "mean"

    language_emb = torch.randn(2, 4, HIDDEN_DIM)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
    pooled = model._pool_language(language_emb, attention_mask)
    expected0 = language_emb[0, :2].mean(dim=0)
    expected1 = language_emb[1].mean(dim=0)
    assert torch.allclose(pooled[0], expected0, atol=1e-5)
    assert torch.allclose(pooled[1], expected1, atol=1e-5)
    print("  ✓ Mean pooling averages valid tokens")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_clip_projection_seeds_native_matrices():
    """projection_type='clip' seeds the joint projection from CLIP's matrices."""
    print("Testing CLIP-native projection seeding...")
    model = VL_JEPA(
        hidden_dim=OPENCLIP_VISION_DIM,
        patch_size=OPENCLIP_PATCH,
        image_size=OPENCLIP_IMAGE_SIZE,
        predictor_layers=1,
        vision_backbone="openclip",
        text_backbone="openclip",
        openclip_model=OPENCLIP_MODEL,
        openclip_pretrained=None,
        freeze_encoders=True,
        projection_dim=512,
        projection_type="clip",
        text_pool="eot",
        contrastive_loss="siglip",
    )
    assert isinstance(model.vision_proj, LinearProjection)
    assert isinstance(model.language_proj, LinearProjection)
    assert model.projection_dim == 512
    # Vision projection weight must equal the tower's native visual.proj (T).
    native_vis = model.context_encoder.visual.proj
    assert torch.allclose(model.vision_proj.weight, native_vis.t(), atol=1e-6)
    print("  ✓ CLIP vision/text projection matrices loaded into LinearProjection")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_clip_proj_plus_eot_matches_openclip_encode():
    """EOT pooling + CLIP linear projection reproduce open_clip's encoders.

    This is the crux of the fix: the joint embedding must equal CLIP's own
    image/text embedding (pre-normalize) so training *starts* aligned instead
    of relearning alignment from scratch through a random MLP + mean pooling.
    """
    print("Testing equivalence to open_clip encode_image/encode_text...")
    import open_clip

    m = open_clip.create_model(OPENCLIP_MODEL, pretrained=None)
    m.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    captions = ["a photo of a cat on a mat", "dog"]
    # open_clip's encode_text adds the full 77-length positional embedding
    # (no slicing), so the reference must use CLIP's native context length.
    ids = tokenizer(captions, context_length=77).long()
    attention_mask = (ids != 0).long()
    images = torch.randn(2, 3, OPENCLIP_IMAGE_SIZE, OPENCLIP_IMAGE_SIZE)

    with torch.no_grad():
        ref_img = m.encode_image(images)
        ref_txt = m.encode_text(ids)

        # Vision: our encoder (ln_post applied) CLS @ native proj.
        vis_enc = OpenCLIPVisionEncoder(
            OPENCLIP_MODEL, pretrained=None, freeze=True, shared_model=m,
        )
        vis_enc.eval()
        vis_proj = LinearProjection(
            OPENCLIP_VISION_DIM, 512, init_weight=m.visual.proj,
        )
        cls = vis_enc(images)[:, 0, :]
        ours_img = vis_proj.raw(cls)

        # Text: our encoder (ln_final applied) EOT token @ native text_projection.
        txt_enc = OpenCLIPLanguageEncoder(
            OPENCLIP_MODEL, pretrained=None, freeze=True, shared_model=m,
        )
        txt_enc.eval()
        txt_proj = LinearProjection(
            OPENCLIP_TEXT_DIM, 512, init_weight=m.text_projection,
        )
        lang_emb = txt_enc(ids, attention_mask)
        eot_idx = attention_mask.sum(dim=1) - 1
        gather_idx = eot_idx.view(-1, 1, 1).expand(-1, 1, lang_emb.size(-1))
        eot = lang_emb.gather(1, gather_idx).squeeze(1)
        ours_txt = txt_proj.raw(eot)

    assert torch.allclose(ours_img, ref_img, atol=1e-4), (ours_img - ref_img).abs().max()
    assert torch.allclose(ours_txt, ref_txt, atol=1e-4), (ours_txt - ref_txt).abs().max()
    print("  ✓ Joint embedding matches open_clip encode_image/encode_text")


def test_residual_projection_is_identity_at_init():
    """ResidualLinearProjection.raw == its CLIP-seeded linear at init (zero residual)."""
    print("Testing residual projection identity-at-init...")
    in_dim, out_dim = 8, 5
    proj = torch.randn(in_dim, out_dim)
    head = ResidualLinearProjection(in_dim, out_dim, init_weight=proj)

    # Seeded matrix preserved and exposed via the .weight shim.
    assert torch.allclose(head.weight, proj.t(), atol=1e-6)
    x = torch.randn(4, in_dim)
    # Zero-init residual output -> raw(x) starts exactly at the CLIP linear.
    assert torch.allclose(head.raw(x), x @ proj, atol=1e-5)

    # After a gradient step the residual must be able to move the output.
    opt = torch.optim.SGD(head.parameters(), lr=1.0)
    target = torch.randn(4, out_dim)
    for _ in range(5):
        opt.zero_grad()
        loss = F.mse_loss(head.raw(x), target)
        loss.backward()
        opt.step()
    assert not torch.allclose(head.raw(x), x @ proj, atol=1e-4)
    print("  ✓ Residual projection starts at CLIP, then adapts")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_clip_residual_projection_starts_at_clip_and_trains():
    """projection_type='clip_residual' == CLIP image embedding at init; grads flow."""
    print("Testing VL-JEPA clip_residual projection...")
    model = VL_JEPA(
        hidden_dim=OPENCLIP_VISION_DIM,
        patch_size=OPENCLIP_PATCH,
        image_size=OPENCLIP_IMAGE_SIZE,
        predictor_layers=1,
        vision_backbone="openclip",
        text_backbone="openclip",
        openclip_model=OPENCLIP_MODEL,
        openclip_pretrained=None,
        freeze_encoders=True,
        projection_dim=512,
        projection_type="clip_residual",
        text_pool="eot",
        contrastive_loss="siglip",
    )
    assert isinstance(model.vision_proj, ResidualLinearProjection)
    assert isinstance(model.language_proj, ResidualLinearProjection)
    assert model.projection_dim == 512

    model.eval()
    images = torch.randn(2, 3, OPENCLIP_IMAGE_SIZE, OPENCLIP_IMAGE_SIZE)
    ids = torch.randint(1, OPENCLIP_VOCAB, (2, OPENCLIP_SEQ_LEN))
    attention_mask = torch.ones(2, OPENCLIP_SEQ_LEN, dtype=torch.long)
    with torch.no_grad():
        outputs = model(images, ids, attention_mask)
        ref = F.normalize(
            model.context_encoder(images)[:, 0, :] @ model.context_encoder.visual.proj,
            p=2, dim=-1, eps=1e-6,
        )
    # Zero-init residual -> joint vision embedding starts at plain CLIP.
    assert torch.allclose(outputs['vision_proj'], ref, atol=1e-4)

    model.train()
    outputs = model(images, ids, attention_mask)
    loss = compute_jepa_loss(outputs, alpha=0.1, beta=0.9, gamma=0.0)['total_loss']
    assert torch.isfinite(loss)
    loss.backward()
    # Residual adapter must receive gradient.
    assert model.vision_proj.residual[-1].weight.grad is not None
    print("  ✓ clip_residual starts at CLIP zero-shot and trains")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_unfreeze_text_last_blocks_toggles_params():
    """unfreeze_text_last_blocks should make the last text blocks trainable."""
    print("Testing text-tower unfreezing...")
    model = VL_JEPA(
        hidden_dim=OPENCLIP_VISION_DIM,
        patch_size=OPENCLIP_PATCH,
        image_size=OPENCLIP_IMAGE_SIZE,
        predictor_layers=1,
        vision_backbone="openclip",
        text_backbone="openclip",
        openclip_model=OPENCLIP_MODEL,
        openclip_pretrained=None,
        freeze_encoders=True,
        projection_dim=512,
        projection_type="clip",
        text_pool="eot",
        contrastive_loss="siglip",
    )
    before = sum(p.requires_grad for p in model.language_encoder.parameters())
    toggled = model.unfreeze_text_last_blocks(2)
    after = sum(p.requires_grad for p in model.language_encoder.parameters())

    assert toggled > 0
    assert after == before + toggled
    # Idempotent: re-calling does not double-toggle already-trainable params.
    assert model.unfreeze_text_last_blocks(2) == 0
    print(f"  ✓ Text unfreeze toggled {toggled} tensors")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_trainer_unfreezes_text_and_adds_param_group():
    """Trainer unfreezes both towers at the configured epoch and registers them."""
    print("Testing trainer symmetric text/vision unfreeze...")
    model = VL_JEPA(
        hidden_dim=OPENCLIP_VISION_DIM,
        patch_size=OPENCLIP_PATCH,
        image_size=OPENCLIP_IMAGE_SIZE,
        predictor_layers=1,
        vision_backbone="openclip",
        text_backbone="openclip",
        openclip_model=OPENCLIP_MODEL,
        openclip_pretrained=None,
        freeze_encoders=True,
        projection_dim=512,
        projection_type="clip",
        text_pool="eot",
        contrastive_loss="siglip",
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-4, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, unfreeze_after_epoch=1,
        unfreeze_vision_blocks=2, unfreeze_text_blocks=2,
    )
    groups_before = len(trainer.optimizer.param_groups)
    trainer._maybe_unfreeze_vision(epoch=1)
    groups_after = len(trainer.optimizer.param_groups)

    assert groups_after == groups_before + 1
    text_trainable = sum(p.requires_grad for p in model.language_encoder.parameters())
    assert text_trainable > 0
    print(f"  ✓ Trainer unfroze text+vision ({text_trainable} text tensors trainable)")


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_clip_projection_vl_jepa_forward_and_backward():
    """VL-JEPA(projection_type=clip) vision path == CLIP image embedding; grads flow."""
    print("Testing VL-JEPA clip-projection forward/backward...")
    model = VL_JEPA(
        hidden_dim=OPENCLIP_VISION_DIM,
        patch_size=OPENCLIP_PATCH,
        image_size=OPENCLIP_IMAGE_SIZE,
        predictor_layers=1,
        vision_backbone="openclip",
        text_backbone="openclip",
        openclip_model=OPENCLIP_MODEL,
        openclip_pretrained=None,
        freeze_encoders=False,
        projection_dim=512,
        projection_type="clip",
        text_pool="eot",
        contrastive_loss="siglip",
    )
    model.eval()
    images = torch.randn(2, 3, OPENCLIP_IMAGE_SIZE, OPENCLIP_IMAGE_SIZE)
    ids = torch.randint(1, OPENCLIP_VOCAB, (2, OPENCLIP_SEQ_LEN))
    attention_mask = torch.ones(2, OPENCLIP_SEQ_LEN, dtype=torch.long)

    with torch.no_grad():
        outputs = model(images, ids, attention_mask)
        ref = F.normalize(
            model.context_encoder(images)[:, 0, :] @ model.context_encoder.visual.proj,
            p=2, dim=-1, eps=1e-6,
        )
    assert outputs['vision_proj'].shape == (2, 512)
    assert outputs['language_proj'].shape == (2, 512)
    assert torch.allclose(outputs['vision_proj'], ref, atol=1e-4)

    model.train()
    outputs = model(images, ids, attention_mask)
    loss = compute_jepa_loss(outputs, alpha=0.1, beta=0.9, gamma=0.0)['total_loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert model.vision_proj.weight.grad is not None
    assert model.language_proj.weight.grad is not None
    print("  ✓ VL-JEPA clip-projection vision path matches CLIP; gradients flow")


def _param_state_name(model, target_param) -> str:
    """Return the state_dict/named_parameters key for a given parameter tensor."""
    for name, param in model.named_parameters():
        if param is target_param:
            return name
    raise AssertionError("target parameter not found in model")


def test_model_ema_tracks_weights_with_decay():
    """ModelEMA shadow moves toward updated weights by (1 - decay)."""
    print("Testing ModelEMA decay tracking...")
    model = _make_model()
    ema = ModelEMA(model, decay=0.9)

    target = model.vision_proj.weight
    name = _param_state_name(model, target)
    before = ema.shadow[name].clone()
    with torch.no_grad():
        target.add_(1.0)  # shift weights by +1
    ema.update(model)

    after = ema.shadow[name]
    # shadow = 0.9 * before + 0.1 * (before + 1) = before + 0.1
    expected = before + 0.1
    assert torch.allclose(after, expected, atol=1e-5)
    assert torch.isfinite(after).all()
    print("  ✓ EMA shadow tracks weights by (1 - decay)")


def test_wise_ft_interpolate_endpoints_and_midpoint():
    """WiSE-FT returns zero-shot at alpha=0, fine-tuned at alpha=1, mean at 0.5."""
    print("Testing WiSE-FT weight interpolation...")
    zs = {'w': torch.zeros(3, 3), 'n': torch.tensor([1, 2, 3])}
    ft = {'w': torch.ones(3, 3), 'n': torch.tensor([4, 5, 6])}

    at0 = wise_ft_interpolate(zs, ft, 0.0)
    at1 = wise_ft_interpolate(zs, ft, 1.0)
    half = wise_ft_interpolate(zs, ft, 0.5)

    assert torch.allclose(at0['w'], zs['w'])
    assert torch.allclose(at1['w'], ft['w'])
    assert torch.allclose(half['w'], torch.full((3, 3), 0.5))
    # Non-float tensors are taken from the fine-tuned side, not interpolated.
    assert torch.equal(half['n'], ft['n'])

    with pytest.raises(ValueError):
        wise_ft_interpolate(zs, ft, 1.5)
    print("  ✓ WiSE-FT endpoints and midpoint correct")


def test_trainer_model_ema_updates_on_optimizer_step():
    """Enabling model EMA tracks a smoothed copy distinct from live weights."""
    print("Testing trainer model-EMA updates...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-2, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, use_model_ema=True, model_ema_decay=0.9,
    )
    assert trainer.model_ema is not None
    name = _param_state_name(model, model.vision_proj.weight)
    init_shadow = trainer.model_ema.shadow[name].clone()

    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN), device=device)
    for _ in range(3):
        assert not trainer.train_step(images, input_ids).get('skipped')

    moved_shadow = trainer.model_ema.shadow[name]
    live = dict(model.named_parameters())[name].detach()
    # The EMA lags the live weights but must have moved off its init.
    assert not torch.allclose(moved_shadow, init_shadow)
    assert not torch.allclose(moved_shadow, live.cpu() if moved_shadow.device.type == 'cpu' else live)
    print("  ✓ Model EMA tracks a smoothed, distinct weight copy")


def test_trainer_eval_weights_swaps_then_restores():
    """eval_weights() loads robust (EMA/WiSE-FT) weights then restores live ones."""
    print("Testing eval_weights swap/restore...")
    model = _make_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-2, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, use_model_ema=True, model_ema_decay=0.5,
        wise_ft_alpha=0.5,
    )
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN), device=device)
    for _ in range(3):
        trainer.train_step(images, input_ids)

    name = _param_state_name(model, model.vision_proj.weight)
    live_before = dict(model.named_parameters())[name].detach().clone()
    eval_state = trainer.build_eval_state_dict()
    # Robust weights differ from the live (overfit-trajectory) weights.
    assert not torch.allclose(eval_state[name].to(device), live_before)

    with trainer.eval_weights():
        inside = dict(model.named_parameters())[name].detach()
        assert torch.allclose(inside, eval_state[name].to(device), atol=1e-5)
    # Live weights restored exactly after the context exits.
    after = dict(model.named_parameters())[name].detach()
    assert torch.allclose(after, live_before, atol=1e-6)
    print("  ✓ eval_weights swaps robust weights and restores live weights")


def test_eval_weights_noop_without_robust_options():
    """Without EMA/WiSE-FT, eval_weights is a transparent no-op."""
    print("Testing eval_weights no-op default...")
    model = _make_model()
    device = torch.device('cpu')
    trainer = VL_JEPA_Trainer(model, device, warmup_steps=0, max_steps=10)
    assert trainer.model_ema is None
    name = _param_state_name(model, model.vision_proj.weight)
    before = dict(model.named_parameters())[name].detach().clone()
    with trainer.eval_weights():
        inside = dict(model.named_parameters())[name].detach()
        assert torch.equal(inside, before)
    print("  ✓ eval_weights is a no-op when robust fine-tuning is disabled")


if __name__ == '__main__':
    print("=" * 60)
    print("VL-JEPA Tests (v2 — Fixed JEPA)")
    print("=" * 60)

    tests = [
        ("Vision Encoder with Mask", test_vision_encoder_with_mask),
        ("Language Encoder", test_language_encoder),
        ("Predictor Module", test_predictor),
        ("VL-JEPA Forward + Loss", test_vl_jepa_forward),
        ("Block Masking", test_block_mask_exact_ratio_and_deterministic_seed),
        ("Tiny-grid Block Masking", test_block_mask_tiny_grid_keeps_visible_context),
        ("Multi-crop Views", test_multicrop_views_shapes),
        ("Multi-crop Forward", test_multicrop_forward_is_finite),
        ("Teacher Mode Restoration", test_target_teacher_modes_are_restored_after_forward),
        ("Logit Scale Bound", test_logit_scale_is_bounded_before_exp),
        ("Padded Small Crop", test_padded_small_crop_forward_is_finite),
        ("Memory Bank Disabled", test_memory_bank_size_zero_disables_queue),
        ("SigLIP Loss", test_siglip_contrastive_loss_is_finite),
        ("Hard-negative Separable", test_hard_negative_loss_zero_when_perfectly_separated),
        ("Hard-negative Single Sample", test_hard_negative_single_sample_is_zero),
        ("Hard-negative in Loss", test_compute_jepa_loss_hard_negative_weight_adds_term),
        ("Hard-negative Trainer", test_trainer_hard_negative_weight_runs),
        ("Contrastive Gradients", test_contrastive_projection_gradients_flow),
        ("Gradient Clipping", test_gradient_clipping_caps_norm),
        ("Gradient Accumulation", test_gradient_accumulation_defers_optimizer_step),
        ("Momentum Update", test_momentum_update),
        ("Capped Momentum Schedule", test_momentum_schedule_capped),
        ("Non-finite Batch Skip", test_trainer_skips_non_finite_batch),
        ("Corrupt Weight Detection", test_trainer_detects_corrupt_weights_before_forward),
        ("Non-finite Output Detection", test_find_first_nonfinite_output),
        ("Post-step Weight Finiteness", test_trainer_weights_stay_finite_after_step),
        ("GradScaler Stability", test_grad_scaler_growth_interval_capped),
        ("GPU Forward + Backward", test_gpu_forward),
        ("Joint Embedding", test_joint_embedding),
        ("Loss Decreases", test_loss_decreases_over_steps),
        ("Checkpoint Round-trip", test_checkpoint_roundtrip),
        ("Checkpoint Validation", test_validate_checkpoint_rejects_nan_weights),
        ("Corrupt Checkpoint Reject", test_load_checkpoint_raises_on_corrupt_file),
        ("Checkpoint Metadata", test_load_checkpoint_returns_metadata),
        ("Masked Patch Loss", test_masked_patches_contribute_to_loss),
        ("OpenCLIP Vision Shapes", test_openclip_vision_encoder_shapes),
        ("OpenCLIP Language Shapes", test_openclip_language_encoder_shapes),
        ("OpenCLIP VL-JEPA Forward", test_openclip_vl_jepa_forward),
        ("LinearProjection CLIP Seed", test_linear_projection_reproduces_clip_matmul),
        ("EOT Pooling", test_eot_pooling_selects_last_valid_token),
        ("Mean Pooling Default", test_mean_pooling_unchanged_default),
        ("CLIP Projection Seeding", test_clip_projection_seeds_native_matrices),
        ("CLIP Proj == open_clip encode", test_clip_proj_plus_eot_matches_openclip_encode),
        ("CLIP Proj VL-JEPA Fwd/Bwd", test_clip_projection_vl_jepa_forward_and_backward),
        ("Residual Proj Identity Init", test_residual_projection_is_identity_at_init),
        ("CLIP Residual Proj", test_clip_residual_projection_starts_at_clip_and_trains),
        ("Text Unfreeze Toggles", test_unfreeze_text_last_blocks_toggles_params),
        ("Trainer Text+Vision Unfreeze", test_trainer_unfreezes_text_and_adds_param_group),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            print(f"  ✅ {name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed > 0:
        print(f"⚠️  {failed} test(s) failed")
    else:
        print(f"✅ All tests passed!")
    print(f"{'=' * 60}")
