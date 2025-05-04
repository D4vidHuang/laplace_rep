import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_pth = 'C:/Delft/MDL/laplace_rep/week2/Zeyu/data'

def get_mnist(batch_size=128):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(MNIST_pth, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(MNIST_pth, train=False, transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_ood_mnist(name, batch_size=128):
    transform = transforms.ToTensor()
    name = name.lower()

    if name == 'emnist':
        dataset = datasets.EMNIST(MNIST_pth, split='letters', train=False, transform=transform, download=False)
    elif name == 'fmnist':
        dataset = datasets.FashionMNIST(MNIST_pth, train=False, transform=transform, download=False)
    elif name == 'kmnist':
        dataset = datasets.KMNIST(MNIST_pth, train=False, transform=transform, download=False)
    else:
        raise ValueError(f"Unknown dataset: '{name}'. Choose from: emnist, fmnist, kmnist")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)