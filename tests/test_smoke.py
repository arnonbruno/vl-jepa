"""Smoke tests for VL-JEPA model."""

import torch
import sys
sys.path.insert(0, '/home/ulluboz/.openclaw/workspace/vl-jepa')

from src.model import VL_JEPA, VisionEncoder, LanguageEncoder


def test_vision_encoder():
    """Test vision encoder shape and forward pass."""
    print("Testing VisionEncoder...")
    encoder = VisionEncoder(hidden_dim=768, patch_size=16, image_size=224)
    
    # Random image batch
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    
    # Should be (B, num_patches+1, hidden_dim) = (2, 196+1, 768)
    assert out.shape == (2, 197, 768), f"Expected (2, 197, 768), got {out.shape}"
    print(f"  ✓ Vision output shape: {out.shape}")


def test_language_encoder():
    """Test language encoder shape and forward pass."""
    print("Testing LanguageEncoder...")
    encoder = LanguageEncoder(vocab_size=30522, hidden_dim=768, max_seq_len=512)
    
    # Random token IDs (batch of 2, seq_len 128)
    input_ids = torch.randint(0, 30522, (2, 128))
    out = encoder(input_ids)
    
    # Should be (B, seq_len, hidden_dim) = (2, 128, 768)
    assert out.shape == (2, 128, 768), f"Expected (2, 128, 768), got {out.shape}"
    print(f"  ✓ Language output shape: {out.shape}")


def test_vl_jepa_forward():
    """Test full VL-JEPA forward pass."""
    print("Testing VL-JEPA forward pass...")
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (2, 128))
    
    output = model(images, input_ids)
    
    # Check all output keys
    assert 'vision_emb' in output
    assert 'language_emb' in output
    assert 'vision_proj' in output
    assert 'language_proj' in output
    assert 'vision_logits' in output
    assert 'language_logits' in output
    
    print(f"  ✓ vision_emb shape: {output['vision_emb'].shape}")
    print(f"  ✓ language_emb shape: {output['language_emb'].shape}")
    print(f"  ✓ vision_logits shape: {output['vision_logits'].shape}")
    print(f"  ✓ language_logits shape: {output['language_logits'].shape}")


def test_vl_jepa_gpu():
    """Test VL-JEPA on GPU if available."""
    print("Testing VL-JEPA on GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = VL_JEPA(hidden_dim=768).to(device)
    
    images = torch.randn(2, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 30522, (2, 128)).to(device)
    
    with torch.no_grad():
        output = model(images, input_ids)
    
    print(f"  ✓ Forward pass successful on {device}")
    print(f"  ✓ GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB" if device.type == 'cuda' else "")


def test_joint_embedding():
    """Test joint embedding extraction."""
    print("Testing joint embedding extraction...")
    model = VL_JEPA(hidden_dim=768)
    
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (2, 128))
    
    vision_proj, language_proj = model.get_joint_embedding(images, input_ids)
    
    assert vision_proj.shape == (2, 768), f"Expected (2, 768), got {vision_proj.shape}"
    assert language_proj.shape == (2, 768), f"Expected (2, 768), got {language_proj.shape}"
    
    # Check normalization
    vision_norm = torch.norm(vision_proj, p=2, dim=-1)
    language_norm = torch.norm(language_proj, p=2, dim=-1)
    assert torch.allclose(vision_norm, torch.ones_like(vision_norm), atol=1e-5)
    assert torch.allclose(language_norm, torch.ones_like(language_norm), atol=1e-5)
    
    print(f"  ✓ Vision embedding shape: {vision_proj.shape}")
    print(f"  ✓ Language embedding shape: {language_proj.shape}")
    print(f"  ✓ Embeddings are normalized")


if __name__ == '__main__':
    print("=" * 60)
    print("VL-JEPA Smoke Tests")
    print("=" * 60)
    
    try:
        test_vision_encoder()
        test_language_encoder()
        test_vl_jepa_forward()
        test_vl_jepa_gpu()
        test_joint_embedding()
        
        print("\n" + "=" * 60)
        print("✅ All smoke tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
