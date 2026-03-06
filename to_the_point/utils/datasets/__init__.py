"""Dataset loading utilities for to_the_point.

These require ``torchvision`` which is an optional dependency.
"""


def __getattr__(name):
    _loaders = {
        "load_cifar10_data",
        "load_cifar10_test_data",
        "load_cifar100_data",
        "load_cifar100_test_data",
        "load_mnist_data",
        "load_mnist_test_data",
        "load_imagenet_data",
        "load_imagenet_test_data",
        "load_wikitext2_data",
        "load_wikitext2_test_data",
        "load_squad_data",
        "load_squad_test_data",
    }
    if name in _loaders:
        # Lazy import to avoid hard dependency on torchvision
        if name.startswith("load_cifar10"):
            from .cifar10 import load_cifar10_data, load_cifar10_test_data
            return load_cifar10_data if name == "load_cifar10_data" else load_cifar10_test_data
        elif name.startswith("load_cifar100"):
            from .cifar100 import load_cifar100_data, load_cifar100_test_data
            return load_cifar100_data if name == "load_cifar100_data" else load_cifar100_test_data
        elif name.startswith("load_mnist"):
            from .mnist import load_mnist_data, load_mnist_test_data
            return load_mnist_data if name == "load_mnist_data" else load_mnist_test_data
        elif name.startswith("load_imagenet"):
            from .imagenet import load_imagenet_data, load_imagenet_test_data
            return load_imagenet_data if name == "load_imagenet_data" else load_imagenet_test_data
        elif name.startswith("load_wikitext2"):
            from .wikitext2 import load_wikitext2_data, load_wikitext2_test_data
            return load_wikitext2_data if name == "load_wikitext2_data" else load_wikitext2_test_data
        elif name.startswith("load_squad"):
            from .squad import load_squad_data, load_squad_test_data
            return load_squad_data if name == "load_squad_data" else load_squad_test_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_cifar10_data",
    "load_cifar10_test_data",
    "load_cifar100_data",
    "load_cifar100_test_data",
    "load_mnist_data",
    "load_mnist_test_data",
    "load_imagenet_data",
    "load_imagenet_test_data",
    "load_wikitext2_data",
    "load_wikitext2_test_data",
    "load_squad_data",
    "load_squad_test_data",
]
