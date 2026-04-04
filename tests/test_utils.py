"""Unit tests for utility helpers."""

import pytest
import torch
from to_the_point import torch_to_analytical
from to_the_point.layers import Model, Linear


def test_torch_to_analytical_forward():
    relu_layer = torch_to_analytical(torch.relu)
    x = torch.randn(5, 4)
    out = relu_layer(x)
    assert (out >= 0).all()
    assert out.shape == x.shape


def test_torch_to_analytical_fit_batch_noop():
    layer = torch_to_analytical(torch.sigmoid)
    layer.fit_batch(torch.randn(5, 4), torch.randn(5, 4))


def test_torch_to_analytical_finalize_fit_noop():
    layer = torch_to_analytical(torch.tanh)
    layer.finalize_fit()


def test_torch_to_analytical_in_model():
    torch.manual_seed(0)
    model = Model(
        Linear(8, 4),
        torch_to_analytical(torch.relu),
        Linear(4, 2),
    )
    X = torch.randn(20, 8)
    Y = torch.randn(20, 2)
    model.fit(X, Y, verbosity=False)
    out = model(X)
    assert out.shape == (20, 2)
