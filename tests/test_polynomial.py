"""Tests for the Polynomial analytical layer."""

import torch
import pytest
from to_the_point import Polynomial


class TestPolynomialForward:
    def test_fit_and_forward(self):
        torch.manual_seed(0)
        in_f, out_f, n = 8, 4, 200
        X = torch.randn(n, in_f)
        W = torch.randn(in_f, out_f)
        Y = X @ W  # linear relationship (degree-1)

        layer = Polynomial(
            in_features=in_f,
            out_features=out_f,
            n_degree=2,
            n_components=8,
            max_cross_terms=20,
            use_cross_terms=False,
        )
        layer.fit_batch(X, Y)
        layer.finalize_fit()

        preds = layer.forward(X)
        mse = (preds - Y).pow(2).mean().item()
        # Polynomial with degree 2 should capture a linear relationship
        assert mse < 1.0, f"MSE too high: {mse}"

    def test_feature_dim_positive(self):
        layer = Polynomial(
            in_features=16, out_features=4, n_degree=2, n_components=8
        )
        assert layer.feature_dim > 0

    def test_not_fitted_raises(self):
        layer = Polynomial(in_features=8, out_features=4, n_components=8)
        with pytest.raises(RuntimeError, match="not fitted"):
            layer.forward(torch.randn(4, 8))
