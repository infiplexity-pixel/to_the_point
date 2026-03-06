import torchvision
import torchvision.transforms as transforms
import torch


def load_cifar100_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    trainset = list(
        torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
    )

    X = torch.stack([x for x, y in trainset]).reshape(-1, 3, 32, 32)
    N = X.shape[0]
    Y = torch.zeros(N, 100)
    for i, (x, y) in enumerate(trainset):
        Y[i, y] = 1
    return X, Y


def load_cifar100_test_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    trainset = list(
        torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    )

    X = torch.stack([x for x, y in trainset]).reshape(-1, 3, 32, 32)
    N = X.shape[0]
    Y = torch.zeros(N, 100)
    for i, (x, y) in enumerate(trainset):
        Y[i, y] = 1
    return X, Y
