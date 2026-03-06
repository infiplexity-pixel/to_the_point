"""Tests for the Conv2d analytical layer."""

import torch
import pytest
from to_the_point import Conv2d


class TestConv2dForward:
    def test_output_shape(self):
        layer = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 8)
        out = layer.forward(x)
        assert out.shape == (2, 8, 8, 8)

    def test_output_shape_no_padding(self):
        layer = Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        x = torch.randn(2, 1, 8, 8)
        out = layer.forward(x)
        assert out.shape == (2, 4, 6, 6)


class TestConv2dFit:
    def test_fit_batch_and_finalize(self):
        layer = Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        X = torch.randn(8, 1, 8, 8)
        Y = torch.zeros(8, 4)
        Y[:, 0] = 1.0  # all same class
        layer.fit_batch(X, Y)
        layer.finalize_fit()
        # After fitting, forward should still work
        out = layer.forward(X)
        assert out.shape == (8, 4, 8, 8)

    def test_empty_batch_ignored(self):
        layer = Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        layer.fit_batch(torch.empty(0, 1, 8, 8), torch.empty(0, 4))
        assert layer.sample_count == 0
