import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import get_mnist
from utils.models import MLP, LeNet
import os

def train(model, dataloader, epochs=10, lr=1e-3):
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


if __name__ == '__main__':
    #model_name = 'mlp'
    model_name = 'lenet'
    save_pth = 'C:/Delft/MDL/laplace_rep/week2/Zeyu/models/map.pt'
    batch_size = 128
    train_loader, _ = get_mnist(batch_size=batch_size)
    model = MLP() if model_name == 'mlp' else LeNet()
    model = train(model, train_loader)
    torch.save(model.state_dict(), save_pth)