"""Unit tests for the Linear analytical layer."""

import pytest
import torch
from to_the_point import Linear


def test_linear_forward_shape():
    layer = Linear(16, 8)
    x = torch.randn(10, 16)
    out = layer(x)
    assert out.shape == (10, 8)


def test_linear_fit_batch_and_finalize():
    torch.manual_seed(0)
    layer = Linear(4, 2)
    X = torch.randn(20, 4)
    Y = torch.randn(20, 2)
    layer.fit_batch(X, Y)
    layer.finalize_fit()
    assert layer.is_fitted
    assert layer.weight.shape == (4, 2)
    assert layer.bias.shape == (1, 2)


def test_linear_fit_reduces_loss():
    torch.manual_seed(42)
    in_f, out_f = 8, 4
    layer = Linear(in_f, out_f)

    X = torch.randn(100, in_f)
    W_true = torch.randn(in_f, out_f)
    Y = X @ W_true + 0.01 * torch.randn(100, out_f)

    layer.fit_batch(X, Y)
    layer.finalize_fit()

    with torch.no_grad():
        pred = layer(X)
        loss = torch.nn.functional.mse_loss(pred, Y).item()

    assert loss < 1.0, f"Loss {loss} is unexpectedly high"


def test_linear_multiple_batches():
    torch.manual_seed(7)
    layer = Linear(4, 2)
    for _ in range(5):
        X = torch.randn(10, 4)
        Y = torch.randn(10, 2)
        layer.fit_batch(X, Y)
    layer.finalize_fit()
    assert layer.is_fitted


def test_linear_repr():
    layer = Linear(10, 5)
    assert repr(layer) == "Linear(10, 5)"


def test_linear_empty_batch_ignored():
    layer = Linear(4, 2)
    X = torch.zeros(0, 4)
    Y = torch.zeros(0, 2)
    layer.fit_batch(X, Y)
    assert layer.sample_count == 0
