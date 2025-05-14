import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import get_mnist
from utils.models import MLP, LeNet
import os
import math

def cosine_annealing_lr(epoch, total_epochs, initial_lr):
    """余弦退火学习率调度"""
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

def train(model, dataloader, epochs=100, initial_lr=0.1, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    # 使用Adam优化器，符合论文描述
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # 更新学习率
        current_lr = cosine_annealing_lr(epoch, epochs, initial_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
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
        print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")

    return model


if __name__ == '__main__':
    # 使用LeNet架构，符合论文描述
    model_name = 'lenet'
    save_dir = 'models'
    save_file = 'MNIST_map.pt'
    save_pth = os.path.join(save_dir, save_file)

    os.makedirs(save_dir, exist_ok=True)
    
   
    batch_size = 128
    # 这里我按照它设置的，但是我觉得不对，我先改5个epoch把后面的做了
    epochs = 5
    initial_lr = 0.1
    weight_decay = 5e-4
    
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    model = LeNet()  # 固定使用LeNet
    
    print(f"Training LeNet on MNIST for {epochs} epochs...")
    print(f"Initial learning rate: {initial_lr}, Weight decay: {weight_decay}")
    model = train(model, train_loader, epochs=epochs, initial_lr=initial_lr, weight_decay=weight_decay)
    
    # 评估模型
    model.eval()
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    torch.save(model.state_dict(), save_pth)
    print(f"Model saved to {save_pth}")