"""Dataset integration tests.

These tests download real datasets (MNIST, CIFAR-10, CIFAR-100), train a
small analytical model, and assert above-chance accuracy.  They are marked
with ``@pytest.mark.dataset`` so they can be skipped during normal unit-test
runs:

    pytest tests/test_datasets.py -v -m dataset
"""

import pytest
import torch
from to_the_point import Linear, Model, Flatten


def _accuracy(model, X, Y):
    """Return top-1 accuracy for a one-hot Y matrix."""
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    labels = Y.argmax(dim=1)
    return (preds == labels).float().mean().item()


@pytest.mark.dataset
class TestMNIST:
    def test_above_chance(self):
        from to_the_point.utils.datasets.mnist import (
            load_mnist_data,
            load_mnist_test_data,
        )

        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)

        model = Model(Linear(784, 10))
        model.fit(X_train, Y_train, batch_size=512, verbosity=False)

        acc = _accuracy(model, X_test, Y_test)
        assert acc > 0.10, f"MNIST accuracy {acc:.3f} is not above chance (10%)"


@pytest.mark.dataset
class TestCIFAR10:
    def test_above_chance(self):
        from to_the_point.utils.datasets.cifar10 import (
            load_cifar10_data,
            load_cifar10_test_data,
        )

        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)

        model = Model(Linear(3 * 32 * 32, 10))
        model.fit(X_train, Y_train, batch_size=512, verbosity=False)

        acc = _accuracy(model, X_test, Y_test)
        assert acc > 0.10, f"CIFAR-10 accuracy {acc:.3f} is not above chance (10%)"


@pytest.mark.dataset
class TestCIFAR100:
    def test_above_chance(self):
        from to_the_point.utils.datasets.cifar100 import (
            load_cifar100_data,
            load_cifar100_test_data,
        )

        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        model = Model(Linear(3 * 32 * 32, 100))
        model.fit(X_train, Y_train, batch_size=512, verbosity=False)

        acc = _accuracy(model, X_test, Y_test)
        assert acc > 0.01, f"CIFAR-100 accuracy {acc:.3f} is not above chance (1%)"
