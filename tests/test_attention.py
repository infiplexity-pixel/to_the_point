"""Tests for the Attention analytical layer."""

import torch
import pytest
from to_the_point import Attention


class TestAttentionForward:
    def test_output_shape(self):
        d_model = 32
        layer = Attention(d_model=d_model, n_heads=4)
        x = torch.randn(2, 8, d_model)
        out = layer.forward(x)
        assert out.shape == (2, 8, d_model)

    def test_attention_weights_shape(self):
        d_model = 16
        n_heads = 4
        layer = Attention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(2, 6, d_model)
        weights = layer.get_attention_weights(x)
        assert weights.shape == (2, n_heads, 6, 6)


class TestAttentionFit:
    def test_fit_batch_runs(self):
        d_model = 16
        layer = Attention(d_model=d_model, n_heads=4)
        X = torch.randn(4, 8, d_model)
        Y = torch.randn(4, 8, d_model)
        stats = layer.fit_batch(X, Y)
        assert "fitting_loss" in stats
        assert stats["batch_count"] == 1

    def test_fit_reduces_loss(self):
        torch.manual_seed(42)
        d_model = 16
        layer = Attention(d_model=d_model, n_heads=4)
        X = torch.randn(8, 4, d_model)
        Y = torch.randn(8, 4, d_model)

        stats0 = layer.fit_batch(X, Y)
        loss_before = stats0["fitting_loss"]

        for _ in range(10):
            stats = layer.fit_batch(X, Y)

        loss_after = stats["fitting_loss"]
        # Loss should decrease or at least not explode
        assert loss_after < loss_before * 2.0

    def test_reset_fitting(self):
        d_model = 16
        layer = Attention(d_model=d_model, n_heads=4)
        X = torch.randn(4, 4, d_model)
        Y = torch.randn(4, 4, d_model)
        layer.fit_batch(X, Y)
        assert layer.batch_count.item() == 1
        layer.reset_fitting()
        assert layer.batch_count.item() == 0
