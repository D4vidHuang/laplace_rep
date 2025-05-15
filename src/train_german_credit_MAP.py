import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import get_german_credit
from utils.models import GermanCreditMLP
import os
import argparse

def train(model, dataloader, epochs=50, lr=1e-3):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on German Credit dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer size')
    args = parser.parse_args()
    
    # 确保保存目录存在
    save_dir = 'models'
    save_file = 'german_credit_map.pt'
    save_pth = os.path.join(save_dir, save_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = get_german_credit(batch_size=args.batch_size)
    
    # 打印数据集信息
    # 获取特征维度
    for x, _ in train_loader:
        input_size = x.size(1)
        print(f"Feature dimension: {input_size}")
        break
        
    # 创建模型
    model = GermanCreditMLP(input_size=input_size, hidden_size=args.hidden_size)
    
    # 训练模型
    model = train(model, train_loader, epochs=args.epochs, lr=args.lr)
    
    # 评估模型
    acc = eval_model(model, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), save_pth)
    print(f"Model saved to {save_pth}") 