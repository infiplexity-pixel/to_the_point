"""
Analytical Conv2d on synthetic image data.

Demonstrates fitting a single Conv2d layer on randomly-generated
image patches and verifying that the forward pass produces the
expected output shape.
"""

import torch
from to_the_point import Conv2d


if __name__ == "__main__":
    torch.manual_seed(0)

    # Synthetic images: 64 samples, 1 channel, 16x16
    n_samples, in_channels, H, W = 64, 1, 16, 16
    n_classes = 10

    X = torch.randn(n_samples, in_channels, H, W)
    labels = torch.randint(0, n_classes, (n_samples,))
    Y = torch.zeros(n_samples, n_classes)
    Y[torch.arange(n_samples), labels] = 1.0

    layer = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, gamma=1e2)

    # Fit analytically
    layer.fit_batch(X, Y)
    layer.finalize_fit()

    # Forward pass
    out = layer(X)
    print(f"Input  shape: {X.shape}")
    print(f"Output shape: {out.shape}")  # expected (64, 8, 16, 16)
