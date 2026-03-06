"""Utility functions for to_the_point."""

from .converter import torch_to_analytical

# Dataset loaders are available via ``to_the_point.utils.datasets`` but
# lazily imported here so the core library works without torchvision.

_DATASET_NAMES = {
    "load_mnist_data",
    "load_mnist_test_data",
    "load_cifar10_data",
    "load_cifar10_test_data",
    "load_cifar100_data",
    "load_cifar100_test_data",
}


def __getattr__(name):
    if name in _DATASET_NAMES:
        from . import datasets
        return getattr(datasets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "torch_to_analytical",
    "load_mnist_data",
    "load_mnist_test_data",
    "load_cifar10_data",
    "load_cifar10_test_data",
    "load_cifar100_data",
    "load_cifar100_test_data",
]
