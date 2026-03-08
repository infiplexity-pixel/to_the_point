"""Unit tests for Model, Residual, Dense, and Flatten."""

import pytest
import torch
from to_the_point import Linear, Model, Flatten
from to_the_point.layers import Dense, Residual


def test_model_forward_shape():
    model = Model(Flatten(), Linear(16, 4))
    x = torch.randn(5, 1, 4, 4)
    out = model(x)
    assert out.shape == (5, 4)


def test_model_fit_synthetic():
    torch.manual_seed(0)
    model = Model(
        Flatten(),
        Linear(16, 8),
        Linear(8, 4),
    )
    X = torch.randn(50, 1, 4, 4)
    Y = torch.randn(50, 4)
    model.fit(X, Y, batch_size=25, verbosity=False)

    out = model(X)
    assert out.shape == (50, 4)


def test_model_batched_forward():
    torch.manual_seed(1)
    model = Model(Flatten(), Linear(16, 4))
    X = torch.randn(20, 1, 4, 4)
    Y = torch.randn(20, 4)
    model.fit(X, Y, verbosity=False)
    out = model.batched_forward(X, batch_size=5)
    assert out.shape == (20, 4)


def test_flatten_shape():
    layer = Flatten()
    x = torch.randn(3, 2, 4)
    out = layer(x)
    assert out.shape == (3, 8)


def test_dense_forward_shape():
    layer = Dense(8, 4)
    x = torch.randn(5, 8)
    out = layer(x)
    assert out.shape == (5, 4)
    assert (out >= 0).all(), "Dense output should be non-negative (ReLU)"


def test_residual_forward_shape():
    model = Residual(Linear(8, 8))
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 8)
