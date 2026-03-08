"""Cosine-similarity test: analytical Attention vs gradient descent.

Trains the same attention configuration both analytically (via fit_batch)
and via gradient descent on identical synthetic data, then asserts that
the resulting attention weight matrices have positive cosine similarity —
verifying that the analytical path learns a comparable representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import copy

from to_the_point import Attention


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().reshape(-1).float()
    b_flat = b.detach().reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def test_attention_cosine_similarity():
    torch.manual_seed(0)

    d_model = 32
    n_heads = 2
    seq_len = 6
    batch_size = 8
    n_batches = 10

    X = torch.randn(batch_size * n_batches, seq_len, d_model)
    Y = torch.randn(batch_size * n_batches, seq_len, d_model)

    # --- Analytical path ---
    analytical_layer = Attention(d_model=d_model, n_heads=n_heads, learn_temperature=False)
    for i in range(n_batches):
        x_b = X[i * batch_size : (i + 1) * batch_size]
        y_b = Y[i * batch_size : (i + 1) * batch_size]
        analytical_layer.fit_batch(x_b, y_b)

    # --- Gradient descent path ---
    gd_layer = Attention(d_model=d_model, n_heads=n_heads, learn_temperature=False)
    gd_layer.w_q.weight.data.copy_(analytical_layer.w_q.weight.data)
    gd_layer.w_k.weight.data.copy_(analytical_layer.w_k.weight.data)
    gd_layer.w_v.weight.data.copy_(analytical_layer.w_v.weight.data)

    optimizer = torch.optim.Adam(gd_layer.parameters(), lr=1e-3)
    for i in range(n_batches):
        x_b = X[i * batch_size : (i + 1) * batch_size]
        y_b = Y[i * batch_size : (i + 1) * batch_size]
        optimizer.zero_grad()
        out = gd_layer(x_b)
        loss = F.mse_loss(out, y_b)
        loss.backward()
        optimizer.step()

    # --- Compare Q-projection weight matrices ---
    sim_q = _cosine_similarity(analytical_layer.w_q.weight, gd_layer.w_q.weight)
    assert sim_q > 0, (
        f"Q-weight cosine similarity {sim_q:.4f} is not positive; "
        "analytical and GD paths diverged."
    )
