import torchvision.transforms as transforms
import torchvision
import torch


def load_cifar10_data(flatten=False):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = list(
        torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    )

    X = torch.stack([x for x, y in trainset]).reshape(-1, 3, 32, 32)
    if flatten:
        X = X.reshape(-1, 3 * 32 * 32)
    N = X.shape[0]
    Y = torch.zeros(N, 10)
    for i, (x, y) in enumerate(trainset):
        Y[i, y] = 1
    return X, Y


def load_cifar10_test_data(flatten=False):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = list(
        torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    )

    X = torch.stack([x for x, y in testset]).reshape(-1, 3, 32, 32)
    if flatten:
        X = X.reshape(-1, 3 * 32 * 32)
    N = X.shape[0]
    Y = torch.zeros(N, 10)
    for i, (x, y) in enumerate(testset):
        Y[i, y] = 1
    return X, Y
