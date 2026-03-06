import torchvision
import torchvision.transforms as transforms
import torch


def load_mnist_data(flatten=False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trainset = list(
        torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    )

    X = torch.stack([x for x, y in trainset]).reshape(-1, 1, 28, 28)
    if flatten:
        X = X.reshape(-1, 28 * 28)
    N = X.shape[0]
    Y = torch.zeros(N, 10)
    for i, (x, y) in enumerate(trainset):
        Y[i, y] = 1
    return X, Y


def load_mnist_test_data(flatten=False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trainset = list(
        torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    )

    X = torch.stack([x for x, y in trainset]).reshape(-1, 1, 28, 28)
    if flatten:
        X = X.reshape(-1, 28 * 28)
    N = X.shape[0]
    Y = torch.zeros(N, 10)
    for i, (x, y) in enumerate(trainset):
        Y[i, y] = 1
    return X, Y
