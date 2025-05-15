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

def train_single_model(model, dataloader, epochs=100, initial_lr=0.1, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
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

def evaluate_ensemble(models, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    
    for models_item in models:
        models_item.eval()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # 收集所有模型的预测
            ensemble_outputs = torch.zeros(len(models), x.size(0), 10).to(device)  # 10是类别数
            for i, model in enumerate(models):
                ensemble_outputs[i] = model(x)
            
            # 平均所有模型的预测
            mean_outputs = ensemble_outputs.mean(dim=0)
            _, predicted = torch.max(mean_outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return 100 * correct / total

if __name__ == '__main__':
    # 配置参数
    model_name = 'lenet'
    save_dir = 'models'
    num_models = 5  # Deep Ensemble中的模型数量
    
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = 128
    epochs = 5
    initial_lr = 0.1
    weight_decay = 5e-4
    
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    
    # 训练5个独立的模型
    models = []
    for i in range(num_models):
        print(f"\nTraining model {i+1}/{num_models}")
        model = LeNet()  # 每次创建新的模型实例
        model = train_single_model(model, train_loader, epochs=epochs, 
                                 initial_lr=initial_lr, weight_decay=weight_decay)
        models.append(model)
        
        # 保存每个模型
        save_path = os.path.join(save_dir, f'MNIST_de_model_{i+1}.pt')
        torch.save(model.state_dict(), save_path)
        print(f"Model {i+1} saved to {save_path}")
    
    # 评估整个集成
    ensemble_accuracy = evaluate_ensemble(models, test_loader)
    print(f'\nEnsemble Test Accuracy: {ensemble_accuracy:.2f}%') 