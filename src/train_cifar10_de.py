import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import get_cifar10
from utils.models import CIFAR10CNN
import os
import argparse

def train_single_model(model, dataloader, epochs=50, lr=1e-3):
    """训练单个模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        avg_loss = total_loss / total
        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")
        
    return model

def eval_model(model, dataloader):
    """评估单个模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    acc = correct / total * 100
    return acc

def train_ensemble(num_models=5, batch_size=128, epochs=50, lr=1e-3):
    """训练Deep Ensemble"""
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = get_cifar10(batch_size=batch_size)
    
    # 训练多个模型
    for i in range(num_models):
        print(f"\nTraining model {i+1}/{num_models}")
        
        # 创建新模型实例
        model = CIFAR10CNN()
        
        # 训练模型
        model = train_single_model(model, train_loader, epochs=epochs, lr=lr)
        
        # 评估模型
        acc = eval_model(model, test_loader)
        print(f"Model {i+1} Test Accuracy: {acc:.2f}%")
        
        # 保存模型
        save_path = os.path.join(save_dir, f'CIFAR10_de_model_{i+1}.pt')
        torch.save(model.state_dict(), save_path)
        print(f"Model {i+1} saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Ensemble on CIFAR10 dataset')
    parser.add_argument('--num_models', type=int, default=5, help='number of models in ensemble')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    
    train_ensemble(
        num_models=args.num_models,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    ) 