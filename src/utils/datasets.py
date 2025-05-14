import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MNIST_pth = 'data'
GERMAN_CREDIT_pth = 'data/GermanCredit'
CIFAR10_pth = 'data'

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


def get_cifar10(batch_size=128):
    """加载CIFAR10数据集
    
    Args:
        batch_size: 批处理大小
        
    Returns:
        train_loader, test_loader
    """
    # 定义数据变换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(CIFAR10_pth, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(CIFAR10_pth, train=False, download=True, transform=transform_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class GermanCreditDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def download_german_credit_dataset():
    """Download German Credit dataset if not exists"""
    import urllib.request
    import os
    
    # 创建目录（如果不存在）
    os.makedirs(GERMAN_CREDIT_pth, exist_ok=True)
    
    # 下载文件
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    file_path = os.path.join(GERMAN_CREDIT_pth, "german.data")
    
    if not os.path.exists(file_path):
        print(f"Downloading German Credit dataset to {file_path}")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete")
    else:
        print(f"German Credit dataset already exists at {file_path}")


def get_german_credit(batch_size=32, test_size=0.2, download=True):
    """Load German Credit dataset
    
    Args:
        batch_size: batch size for DataLoader
        test_size: proportion of test set
        download: whether to download the dataset if not exists
    
    Returns:
        train_loader, test_loader
    """
    if download:
        download_german_credit_dataset()
    
    # 列名
    column_names = [
        'status', 'duration', 'credit_history', 'purpose', 'amount',
        'savings', 'employment_duration', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
        'credit_risk'
    ]
    
    # 加载数据
    file_path = os.path.join(GERMAN_CREDIT_pth, "german.data")
    df = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
    
    # 标签变换: 原始数据中1=好客户, 2=坏客户, 转换为0=好客户, 1=坏客户
    df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
    
    # 分离特征和标签
    X = pd.get_dummies(df.drop('credit_risk', axis=1), drop_first=True).values
    y = df['credit_risk'].values
    
    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 创建 PyTorch 数据集
    train_dataset = GermanCreditDataset(X_train, y_train)
    test_dataset = GermanCreditDataset(X_test, y_test)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_MNIST(batch_size=128, val_size=0.1):
    """Load MNIST dataset with train, validation and test sets
    
    Args:
        batch_size: batch size for DataLoader
        val_size: proportion of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.ToTensor()
    
    # Load the full training set
    full_train_dataset = datasets.MNIST(MNIST_pth, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(MNIST_pth, train=False, transform=transform, download=False)
    
    # Calculate lengths for train and validation
    val_length = int(len(full_train_dataset) * val_size)
    train_length = len(full_train_dataset) - val_length
    
    # Split training dataset into train and validation
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_length, val_length],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader