"""Tests for the converter utility."""

import torch
from to_the_point import torch_to_analytical


class TestTorchToAnalytical:
    def test_wraps_relu(self):
        relu_layer = torch_to_analytical(torch.relu)
        x = torch.randn(4, 8)
        out = relu_layer.forward(x)
        assert (out >= 0).all()
        assert out.shape == x.shape

    def test_has_fit_methods(self):
        layer = torch_to_analytical(torch.sigmoid)
        # These should not raise
        layer.fit_batch()
        layer.finalize_fit()
