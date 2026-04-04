"""
Linear classifier on synthetic data.

Demonstrates the simplest possible use of to_the_point: a single
analytical Linear layer that maps flat inputs to class scores, fitted
in one pass with no optimizer.
"""

import torch
from to_the_point import Linear, Model, Flatten


def make_synthetic_dataset(n_samples=1000, n_features=64, n_classes=10, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    Y = torch.zeros(n_samples, n_classes)
    Y[torch.arange(n_samples), labels] = 1.0
    return X, Y, labels


def accuracy(model, X, labels):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == labels).float().mean().item()


if __name__ == "__main__":
    n_classes = 10
    X_train, Y_train, labels_train = make_synthetic_dataset(n_samples=2000, n_classes=n_classes)
    X_test, Y_test, labels_test = make_synthetic_dataset(n_samples=500, n_classes=n_classes, seed=1)

    # Build and fit the model analytically — no epochs, no optimizer
    model = Model(Linear(64, n_classes))
    model.fit(X_train, Y_train, batch_size=256, verbosity=True)

    train_acc = accuracy(model, X_train, labels_train)
    test_acc = accuracy(model, X_test, labels_test)

    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")
