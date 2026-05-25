"""
Smoke tests for VL-JEPA model (fixed version).
Tests proper JEPA functionality: masking, predictor, loss computation, GPU.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import os

from src.model import (
    VL_JEPA, VisionEncoder, LanguageEncoder, Predictor,
    block_patch_mask, compute_jepa_loss, make_multicrop_views,
)
from src.trainer import VL_JEPA_Trainer

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


def test_multicrop_views_shapes():
    """Global and local crops should preserve batch size and requested resolutions."""
    print("Testing multi-crop view generation...")
    images = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
    views = make_multicrop_views(images, global_size=IMAGE_SIZE, local_size=32, training=True)

    assert views['global'].shape == (3, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert views['local'].shape == (3, 3, 32, 32)
    print(f"  ✓ Global crop: {views['global'].shape}; local crop: {views['local'].shape}")


def test_contrastive_projection_gradients_flow():
    """InfoNCE must update both projection heads."""
    print("Testing contrastive projection gradients...")
    model = _make_model()
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))

    outputs = model(images, input_ids)
    loss = compute_jepa_loss(outputs, alpha=0.0, beta=1.0)['total_loss']
    loss.backward()

    vision_grad = model.vision_proj.weight.grad
    language_grad = model.language_proj.weight.grad
    assert vision_grad is not None and vision_grad.abs().sum() > 0
    assert language_grad is not None and language_grad.abs().sum() > 0
    print("  ✓ vision_proj and language_proj receive InfoNCE gradients")


def test_momentum_update():
    """Test that momentum update works (target != context at init)."""
    print("Testing momentum update...")
    model = _make_model()

    # Before update: should be identical (loaded same weights)
    ctx_state = next(model.context_encoder.parameters())
    tgt_state = next(model.target_encoder.parameters())
    assert torch.allclose(ctx_state, tgt_state), "Target should match context at init"
    lang_state = next(model.language_encoder.parameters())
    tgt_lang_state = next(model.target_language_encoder.parameters())
    assert torch.allclose(lang_state, tgt_lang_state), "Target language should match at init"

    # Modify context
    with torch.no_grad():
        ctx_state.add_(torch.randn_like(ctx_state) * 0.1)
        lang_state.add_(torch.randn_like(lang_state) * 0.1)

    # Momentum update (tau=0.9 for test)
    model.momentum_tau = 0.9
    model.momentum_update()

    # After update: target should have moved partially
    new_tgt = next(model.target_encoder.parameters())
    assert not torch.allclose(ctx_state, new_tgt), "Target should differ from context after update"
    new_tgt_lang = next(model.target_language_encoder.parameters())
    assert not torch.allclose(lang_state, new_tgt_lang), "Target language should EMA-update"
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
    for _ in range(8):
        metrics = trainer.train_step(images, input_ids)
        losses.append(metrics['total_loss'])

    avg_first = sum(losses[:2]) / 2
    avg_last = sum(losses[-2:]) / 2
    assert avg_last < avg_first, (
        f"Expected moving-average loss to decrease: "
        f"first2={avg_first:.4f}, last2={avg_last:.4f}"
    )
    print(f"  ✓ Loss decreased (avg first 2 vs last 2): {avg_first:.4f} → {avg_last:.4f}")


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

    assert loss_masked > 0.01, f"Masked MSE should be > 0, got {loss_masked}"
    assert loss_unmasked < 1e-6, f"Unmasked MSE should be ~0, got {loss_unmasked}"
    print(f"  ✓ MSE (all masked): {loss_masked:.4f}")
    print(f"  ✓ MSE (none masked): {loss_unmasked:.6f}")


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
        ("Multi-crop Views", test_multicrop_views_shapes),
        ("Contrastive Gradients", test_contrastive_projection_gradients_flow),
        ("Momentum Update", test_momentum_update),
        ("GPU Forward + Backward", test_gpu_forward),
        ("Joint Embedding", test_joint_embedding),
        ("Loss Decreases", test_loss_decreases_over_steps),
        ("Checkpoint Round-trip", test_checkpoint_roundtrip),
        ("Masked Patch Loss", test_masked_patches_contribute_to_loss),
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
