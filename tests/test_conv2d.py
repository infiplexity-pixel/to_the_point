"""Unit tests for the Conv2d analytical layer."""

import pytest
import torch
from to_the_point import Conv2d


def test_conv2d_forward_shape():
    layer = Conv2d(1, 4, kernel_size=3, padding=1)
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)


def test_conv2d_output_shape_helper():
    layer = Conv2d(1, 4, kernel_size=3, padding=0)
    h_out, w_out = layer.output_shape((2, 1, 8, 8))
    assert h_out == 6
    assert w_out == 6


def test_conv2d_fit_and_forward():
    torch.manual_seed(0)
    layer = Conv2d(1, 4, kernel_size=3, padding=1, gamma=1e1)
    X = torch.randn(16, 1, 8, 8)
    Y = torch.zeros(16, 10)
    Y[torch.arange(16), torch.randint(0, 10, (16,))] = 1.0

    layer.fit_batch(X, Y)
    layer.finalize_fit()

    out = layer(X)
    assert out.shape == (16, 4, 8, 8)


def test_conv2d_empty_batch_noop():
    layer = Conv2d(1, 4, kernel_size=3, padding=1)
    X = torch.zeros(0, 1, 8, 8)
    Y = torch.zeros(0, 10)
    layer.fit_batch(X, Y)
    assert layer.sample_count == 0


def test_conv2d_finalize_noop_before_fit():
    layer = Conv2d(1, 4, kernel_size=3, padding=1)
    layer.finalize_fit()
