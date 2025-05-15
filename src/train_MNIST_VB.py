import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.datasets import get_mnist
import os
import argparse
from bayesian_torch.layers import Conv2dFlipout, LinearFlipout

class BayesianLeNet(nn.Module):
    def __init__(self, prior_mu=0, prior_sigma=1, posterior_mu_init=0, posterior_rho_init=-3):
        super(BayesianLeNet, self).__init__()
        
        # 设置先验精度为5e-4（转换为标准差）
        prior_sigma = (1.0 / (5e-4))**0.5
        
        # 第一个卷积层块
        self.conv1 = Conv2dFlipout(
            in_channels=1, out_channels=6, kernel_size=5,
            prior_mean=prior_mu, prior_variance=prior_sigma**2,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 第二个卷积层块
        self.conv2 = Conv2dFlipout(
            in_channels=6, out_channels=16, kernel_size=5,
            prior_mean=prior_mu, prior_variance=prior_sigma**2,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 全连接层
        self.fc1 = LinearFlipout(
            in_features=16*4*4, out_features=120,
            prior_mean=prior_mu, prior_variance=prior_sigma**2,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.act3 = nn.ReLU()
        
        self.fc2 = LinearFlipout(
            in_features=120, out_features=84,
            prior_mean=prior_mu, prior_variance=prior_sigma**2,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )
        self.act4 = nn.ReLU()
        
        self.fc3 = LinearFlipout(
            in_features=84, out_features=10,
            prior_mean=prior_mu, prior_variance=prior_sigma**2,
            posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init
        )

    def forward(self, x):
        # Flipout层返回(output, kl)元组，我们只需要output
        x = self.act1(self.conv1(x)[0])
        x = self.pool1(x)
        x = self.act2(self.conv2(x)[0])
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)
        x = self.act3(self.fc1(x)[0])
        x = self.act4(self.fc2(x)[0])
        x = self.fc3(x)[0]
        return x

def train(model, train_loader, epochs=100, lr=0.001, kl_factor=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(x)
            
            # 计算KL散度
            kl = 0
            for module in model.modules():
                if hasattr(module, 'kl_loss'):
                    kl = kl + module.kl_loss()
            
            # 计算总损失（NLL + KL）
            nll = criterion(outputs, y)
            loss = nll + kl_factor * kl
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return model

def evaluate(model, test_loader, num_samples=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # 多次采样预测
            outputs = torch.zeros(num_samples, x.size(0), 10).to(device)
            for i in range(num_samples):
                outputs[i] = model(x)
            
            # 平均预测结果
            mean_output = outputs.mean(dim=0)
            _, predicted = torch.max(mean_output, 1)
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--kl_factor', type=float, default=0.1)
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = get_mnist(batch_size=args.batch_size)
    
    # 创建模型
    model = BayesianLeNet()
    
    # 训练模型
    print("Training Bayesian LeNet...")
    model = train(model, train_loader, epochs=args.epochs, lr=args.lr, kl_factor=args.kl_factor)
    
    # 评估模型
    print("\nEvaluating model...")
    test_accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # 保存模型
    save_path = os.path.join(save_dir, 'MNIST_vb.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}") 