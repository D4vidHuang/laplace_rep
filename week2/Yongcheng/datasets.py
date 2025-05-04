import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class DatasetLoader:
    def __init__(self, data_dir, batch_size=128):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def get_mnist(self, train=True):
        try:
            dataset = datasets.MNIST(
                self.data_dir, train=train, transform=self.transform, download=True
            )
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=train, num_workers=2, pin_memory=True
            )
            return loader
        except Exception as e:
            raise RuntimeError(f"Failed to load MNIST dataset: {str(e)}")

    def get_ood_dataset(self, name):
        name = name.lower()
        try:
            if name == 'emnist':
                dataset = datasets.EMNIST(
                    self.data_dir, split='letters', train=False, transform=self.transform, download=True
                )
            elif name == 'fmnist':
                dataset = datasets.FashionMNIST(
                    self.data_dir, train=False, transform=self.transform, download=True
                )
            elif name == 'kmnist':
                dataset = datasets.KMNIST(
                    self.data_dir, train=False, transform=self.transform, download=True
                )
            else:
                raise ValueError(f"Unknown OOD dataset: {name}. Choose from: emnist, fmnist, kmnist")
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True
            )
            return loader
        except Exception as e:
            raise RuntimeError(f"Failed to load {name} dataset: {str(e)}")