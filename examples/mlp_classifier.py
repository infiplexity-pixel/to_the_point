"""
Multi-layer analytical MLP on synthetic data.

Shows how to stack multiple analytical Linear layers (with an optional
ReLU activation via torch_to_analytical) inside a Model container.
"""

import torch
from to_the_point import Linear, Model, Flatten, torch_to_analytical


def make_synthetic_dataset(n_samples=1000, in_features=128, n_classes=5, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    Y = torch.zeros(n_samples, n_classes)
    Y[torch.arange(n_samples), labels] = 1.0
    return X, Y, labels


def accuracy(model, X, labels):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == labels).float().mean().item()


if __name__ == "__main__":
    in_features, hidden, n_classes = 128, 64, 5

    X_train, Y_train, labels_train = make_synthetic_dataset(
        n_samples=2000, in_features=in_features, n_classes=n_classes
    )
    X_test, Y_test, labels_test = make_synthetic_dataset(
        n_samples=500, in_features=in_features, n_classes=n_classes, seed=99
    )

    # Two-layer MLP: Linear -> ReLU -> Linear
    model = Model(
        Linear(in_features, hidden),
        torch_to_analytical(torch.relu),
        Linear(hidden, n_classes),
    )

    model.fit(X_train, Y_train, batch_size=256, verbosity=True)

    print(f"\nTrain accuracy: {accuracy(model, X_train, labels_train):.3f}")
    print(f"Test  accuracy: {accuracy(model, X_test, labels_test):.3f}")
