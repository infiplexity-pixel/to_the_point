"""Integration test stubs for real-dataset benchmarks.

These tests are designed to run in CI with dataset downloads enabled.
They verify that analytical layers can train on and predict real data
at above-chance accuracy. Mark them with ``@pytest.mark.dataset`` so
they can be selected / excluded easily.
"""

import pytest
import torch

dataset = pytest.mark.dataset


@dataset
class TestMNIST:
    def test_linear_on_mnist(self):
        from to_the_point import Linear
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data

        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)

        X_train = X_train.float()
        X_test = X_test.float()

        layer = Linear(in_features=784, out_features=10)
        layer.fit_batch(X_train, Y_train)
        layer.finalize_fit()

        preds = layer.forward(X_test)
        pred_classes = preds.argmax(dim=1)
        true_classes = Y_test.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean().item()

        assert accuracy > 0.70, f"MNIST accuracy too low: {accuracy:.2%}"


@dataset
class TestCIFAR10:
    def test_linear_on_cifar10(self):
        from to_the_point import Linear
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data

        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)

        X_train = X_train.float()
        X_test = X_test.float()

        layer = Linear(in_features=3072, out_features=10)
        layer.fit_batch(X_train, Y_train)
        layer.finalize_fit()

        preds = layer.forward(X_test)
        pred_classes = preds.argmax(dim=1)
        true_classes = Y_test.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean().item()

        assert accuracy > 0.20, f"CIFAR-10 accuracy too low: {accuracy:.2%}"


@dataset
class TestCIFAR100:
    def test_linear_on_cifar100(self):
        from to_the_point import Linear
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data

        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()

        X_train = X_train.reshape(X_train.shape[0], -1).float()
        X_test = X_test.reshape(X_test.shape[0], -1).float()

        layer = Linear(in_features=3072, out_features=100)
        layer.fit_batch(X_train, Y_train)
        layer.finalize_fit()

        preds = layer.forward(X_test)
        pred_classes = preds.argmax(dim=1)
        true_classes = Y_test.argmax(dim=1)
        accuracy = (pred_classes == true_classes).float().mean().item()

        assert accuracy > 0.05, f"CIFAR-100 accuracy too low: {accuracy:.2%}"
