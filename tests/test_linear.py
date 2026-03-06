"""Tests for the Linear analytical layer."""

import torch
import pytest
from to_the_point import Linear


class TestLinearForward:
    def test_output_shape(self):
        layer = Linear(16, 8)
        x = torch.randn(4, 16)
        out = layer.forward(x)
        assert out.shape == (4, 8)

    def test_batched_forward_large_output(self):
        layer = Linear(16, 512, param_batch_size=128)
        x = torch.randn(4, 16)
        out = layer.forward(x)
        assert out.shape == (4, 512)


class TestLinearFit:
    def test_fit_improves_output(self):
        torch.manual_seed(42)
        in_f, out_f, n = 8, 4, 100
        W_true = torch.randn(in_f, out_f)
        b_true = torch.randn(1, out_f)
        X = torch.randn(n, in_f)
        Y = X @ W_true + b_true

        layer = Linear(in_f, out_f)
        layer.fit_batch(X, Y)
        layer.finalize_fit()

        preds = layer.forward(X)
        mse = (preds - Y).pow(2).mean().item()
        assert mse < 1e-3, f"MSE after fit is too high: {mse}"

    def test_fit_batch_accumulation(self):
        torch.manual_seed(0)
        in_f, out_f, n = 8, 4, 200
        W_true = torch.randn(in_f, out_f)
        X = torch.randn(n, in_f)
        Y = X @ W_true

        layer = Linear(in_f, out_f)
        for start in range(0, n, 50):
            layer.fit_batch(X[start : start + 50], Y[start : start + 50])
        layer.finalize_fit()

        preds = layer.forward(X)
        mse = (preds - Y).pow(2).mean().item()
        assert mse < 1e-3

    def test_zero_batch_ignored(self):
        layer = Linear(4, 2)
        layer.fit_batch(torch.empty(0, 4), torch.empty(0, 2))
        assert layer.sample_count == 0
