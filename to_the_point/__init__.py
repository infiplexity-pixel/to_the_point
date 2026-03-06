"""
to_the_point - Analytical replacements for autograd layers.

A library providing one-to-one analytical replacements for CNNs, Transformers,
DNNs, and other neural network components. Each layer can be fitted analytically
using closed-form solutions (ridge regression, covariance methods) instead of
gradient-based backpropagation.
"""

from .layers import (
    Linear,
    Attention,
    Conv2d,
    Embedding,
    Polynomial,
    Model,
    Residual,
    Dense,
    Flatten,
    Recursive,
    UnEmbed,
)

from .utils import torch_to_analytical

__version__ = "0.1.0"

__all__ = [
    "Linear",
    "Attention",
    "Conv2d",
    "Embedding",
    "Polynomial",
    "Model",
    "Residual",
    "Dense",
    "Flatten",
    "Recursive",
    "UnEmbed",
    "torch_to_analytical",
]
