from .cifar10 import load_cifar10_data, load_cifar10_test_data
from .cifar100 import load_cifar100_data, load_cifar100_test_data
from .mnist import load_mnist_data, load_mnist_test_data
from .imagenet import load_imagenet_data, load_imagenet_test_data
from .wikitext2 import load_wikitext2_data, load_wikitext2_test_data
from .squad import load_squad_data, load_squad_test_data


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
