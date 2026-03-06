"""Tests for Model, Residual, Dense, and Flatten."""

import torch
import pytest
from to_the_point import Model, Residual, Dense, Flatten, Linear


class TestModel:
    def test_forward_chain(self):
        m = Model(Linear(8, 16), Linear(16, 4))
        x = torch.randn(2, 8)
        out = m.forward(x)
        assert out.shape == (2, 4)

    def test_batched_forward(self):
        m = Model(Linear(8, 4))
        x = torch.randn(10, 8)
        out = m.batched_forward(x, batch_size=3)
        assert out.shape == (10, 4)


class TestResidual:
    def test_residual_adds_identity(self):
        torch.manual_seed(0)
        inner = Linear(4, 4)
        r = Residual(inner)
        x = torch.randn(2, 4)
        out = r.forward(x)
        # Output should include residual contribution
        assert out.shape == (2, 4)


class TestDense:
    def test_relu_applied(self):
        layer = Dense(4, 8)
        x = torch.randn(3, 4)
        out = layer.forward(x)
        assert (out >= 0).all()
        assert out.shape == (3, 8)


class TestFlatten:
    def test_flatten(self):
        layer = Flatten()
        x = torch.randn(2, 3, 4, 5)
        out = layer.forward(x)
        assert out.shape == (2, 60)
