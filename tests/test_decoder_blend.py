import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.blend import BlendNetwork, ImplicitFieldWithBlending
from model.decoder import SharedImplicitDecoder


def test_decoder_output_shape() -> None:
    decoder = SharedImplicitDecoder(
        feat_dim=10,
        num_frequencies=4,
        hidden_dim=32,
        num_layers=4,
    )
    x_rel = torch.randn(5, 3, 3)
    feat = torch.randn(5, 3, 10)
    sdf_i = decoder(x_rel=x_rel, feat=feat)
    assert sdf_i.shape == (5, 3, 1)


def test_blend_weights_sum_to_one() -> None:
    blend = BlendNetwork(
        feat_dim=10,
        num_frequencies=4,
        hidden_dim=16,
        num_layers=3,
        use_confidence=True,
    )
    x_rel = torch.randn(4, 6, 3)
    feat = torch.randn(4, 6, 10)
    confidence = torch.rand(4, 6, 1)
    logits, weights = blend(x_rel=x_rel, feat=feat, confidence=confidence)

    assert logits.shape == (4, 6, 1)
    assert weights.shape == (4, 6, 1)
    sums = weights.sum(dim=1).squeeze(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)


def test_end_to_end_fusion_shape() -> None:
    model = ImplicitFieldWithBlending(
        feat_dim=8,
        posenc_frequencies=3,
        decoder_hidden_dim=32,
        decoder_layers=4,
        blend_hidden_dim=16,
        blend_layers=3,
        use_confidence=False,
    )
    x_world = torch.randn(7, 3)
    x_rel = torch.randn(7, 4, 3).clamp(-1.0, 1.0)
    feat = torch.randn(7, 4, 8)

    out = model(x_world=x_world, x_rel=x_rel, feat=feat, confidence=None)
    assert out["sdf_i"].shape == (7, 4, 1)
    assert out["weights"].shape == (7, 4, 1)
    assert out["sdf"].shape == (7, 1)

    sums = out["weights"].sum(dim=1).squeeze(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_decoder_output_shape()
    test_blend_weights_sum_to_one()
    test_end_to_end_fusion_shape()
    print("All decoder/blend tests passed.")

