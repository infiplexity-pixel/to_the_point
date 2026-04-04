"""Unit tests for the Attention analytical layer."""

import pytest
import torch
from to_the_point import Attention


def test_attention_forward_shape():
    layer = Attention(d_model=16, n_heads=2)
    x = torch.randn(2, 4, 16)
    out = layer(x)
    assert out.shape == (2, 4, 16)


def test_attention_fit_batch_returns_stats():
    layer = Attention(d_model=16, n_heads=2)
    X = torch.randn(2, 4, 16)
    Y = torch.randn(2, 4, 16)
    stats = layer.fit_batch(X, Y)
    assert "fitting_loss" in stats
    assert "batch_count" in stats


def test_attention_get_weights_shape():
    layer = Attention(d_model=16, n_heads=2)
    x = torch.randn(2, 4, 16)
    weights = layer.get_attention_weights(x)
    assert weights.shape == (2, 2, 4, 4)


def test_attention_get_fitting_summary_before_fit():
    layer = Attention(d_model=16, n_heads=2)
    summary = layer.get_fitting_summary()
    assert summary["batch_count"] == 0


def test_attention_reset_fitting():
    layer = Attention(d_model=16, n_heads=2)
    X = torch.randn(2, 4, 16)
    Y = torch.randn(2, 4, 16)
    layer.fit_batch(X, Y)
    layer.reset_fitting()
    assert layer.batch_count == 0


def test_attention_set_learning_rates():
    layer = Attention(d_model=16, n_heads=2)
    layer.set_learning_rates(q=0.05, k=0.05, v=0.05, temperature=0.001)
    assert layer.learning_rates["q"] == 0.05
    assert layer.learning_rates["k"] == 0.05
    assert layer.learning_rates["v"] == 0.05
    assert layer.learning_rates["temperature"] == 0.001
