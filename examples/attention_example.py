"""
Analytical Attention on synthetic sequence data.

Shows how to create and analytically fit an Attention layer on
randomly-generated token embeddings, then inspect the fitting stats.
"""

import torch
from to_the_point import Attention


if __name__ == "__main__":
    torch.manual_seed(0)

    d_model, n_heads, seq_len, batch_size = 64, 4, 10, 16

    layer = Attention(d_model=d_model, n_heads=n_heads)

    # Simulate a few batches of sequence data
    for step in range(5):
        X = torch.randn(batch_size, seq_len, d_model)
        Y = torch.randn(batch_size, seq_len, d_model)
        stats = layer.fit_batch(X, Y)
        print(f"Step {step + 1}: loss={stats['fitting_loss']:.4f}")

    # Inference
    x_test = torch.randn(4, seq_len, d_model)
    out = layer(x_test)
    print(f"\nInput  shape: {x_test.shape}")
    print(f"Output shape: {out.shape}")  # expected (4, 10, 64)

    # Summary
    summary = layer.get_fitting_summary()
    print(f"\nFitting summary: {summary}")
