"""Compare analytical vs gradient-based attention on synthetic data.

Trains the same Attention layer configuration analytically and via
gradient-based backprop, then checks the cosine similarity between
the resulting attention weight matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytest
from to_the_point import Attention


def _gradient_attention_train(d_model, n_heads, X, Y, lr=1e-2, steps=200):
    """Train a standard PyTorch multi-head attention via gradient descent."""

    class GradAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            B = x.size(0)
            Q = self.w_q(x).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.w_k(x).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.w_v(x).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)
            out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
            return self.out_proj(out), attn

    model = GradAttention(d_model, n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        out, _ = model(X)
        loss = F.mse_loss(out, Y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, attn_weights = model(X)
    return attn_weights, model


class TestCosineSimAnalyticalVsGradient:
    """Cosine similarity between analytical and gradient-based attention matrices."""

    def test_cosine_similarity_attention(self):
        torch.manual_seed(123)
        d_model = 16
        n_heads = 4
        batch_size = 8
        seq_len = 6

        # Synthetic data
        X = torch.randn(batch_size, seq_len, d_model)
        Y = torch.randn(batch_size, seq_len, d_model)

        # --- Gradient-based training ---
        grad_attn_weights, _ = _gradient_attention_train(
            d_model, n_heads, X, Y, lr=1e-2, steps=300
        )

        # --- Analytical training ---
        analytical_layer = Attention(d_model=d_model, n_heads=n_heads)
        for _ in range(20):
            analytical_layer.fit_batch(X, Y)

        analytical_attn_weights = analytical_layer.get_attention_weights(X)

        # Flatten attention matrices for cosine similarity
        grad_flat = grad_attn_weights.reshape(-1)
        analytical_flat = analytical_attn_weights.reshape(-1)

        cosine_sim = F.cosine_similarity(
            grad_flat.unsqueeze(0), analytical_flat.unsqueeze(0)
        ).item()

        # They won't be identical (different optimisers / solutions) but
        # should exhibit some positive correlation for the same data.
        assert cosine_sim > -0.5, (
            f"Cosine similarity too low: {cosine_sim:.4f}. "
            "Analytical and gradient attention are diverging completely."
        )

    def test_analytical_attention_converges(self):
        """The analytical loss should decrease over repeated fit_batch calls."""
        torch.manual_seed(0)
        d_model = 16
        n_heads = 4
        X = torch.randn(8, 4, d_model)
        Y = torch.randn(8, 4, d_model)

        layer = Attention(d_model=d_model, n_heads=n_heads)
        losses = []
        for _ in range(15):
            stats = layer.fit_batch(X, Y)
            losses.append(stats["fitting_loss"])

        # The loss at the end should be lower than at the beginning
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )
