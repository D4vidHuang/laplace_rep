import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet(nn.Module):
    """
    标准LeNet架构，用于MNIST实验
    论文参考: LeCun et al., 1998
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 第一个卷积层，1个输入通道，6个输出通道，5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层，6个输入通道，16个输出通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 第一个卷积块：卷积+ReLU+池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积块：卷积+ReLU+池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平操作
        x = x.view(x.size(0), -1)
        # 全连接层+ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        return self.fc3(x)


class GermanCreditMLP(nn.Module):
    def __init__(self, input_size=61, hidden_size=32, num_classes=2):
        super(GermanCreditMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class SWAG(nn.Module):
    """
    SWAG (Stochastic Weight Averaging Gaussian) implementation
    Based on the paper: https://arxiv.org/abs/1902.02476
    """
    def __init__(self, base_model, max_models=40):
        super(SWAG, self).__init__()
        self.base_model = base_model
        self.max_models = max_models
        self.n_models = 0
        
        # Initialize parameter lists for storage
        self.params = []
        for param in self.base_model.parameters():
            self.params.append(param.data.clone())
        
        # Register buffers for each parameter tensor
        self.n_params = len(list(self.base_model.parameters()))
        for i, param in enumerate(self.base_model.parameters()):
            self.register_buffer(f'mean_{i}', param.data.clone())
            self.register_buffer(f'sq_mean_{i}', param.data.clone() ** 2)
    
    def forward(self, x):
        return self.base_model(x)
    
    def update_parameters(self, model):
        """Update running average of parameters"""
        self.n_models += 1
        n = self.n_models
        
        for i, param in enumerate(model.parameters()):
            mean = getattr(self, f'mean_{i}')
            sq_mean = getattr(self, f'sq_mean_{i}')
            
            if n == 1:
                mean.data.copy_(param.data)
                sq_mean.data.copy_(param.data ** 2)
            else:
                mean.data.mul_((n-1)/n).add_(param.data/n)
                sq_mean.data.mul_((n-1)/n).add_((param.data ** 2)/n)
    
    def sample(self, scale=1.0, diag_only=True):
        """Sample from the SWAG posterior"""
        if diag_only:
            # Only use diagonal covariance
            for i, param in enumerate(self.base_model.parameters()):
                mean = getattr(self, f'mean_{i}')
                sq_mean = getattr(self, f'sq_mean_{i}')
                var = torch.clamp(sq_mean - mean ** 2, 1e-30)
                eps = torch.randn_like(var)
                param.data.copy_(mean + scale * torch.sqrt(var) * eps)
        else:
            # Full covariance version can be implemented here
            raise NotImplementedError("Full covariance sampling not implemented yet")
    
    def get_space(self):
        """Get the space requirements in bytes"""
        space = 0
        for i in range(self.n_params):
            mean = getattr(self, f'mean_{i}')
            sq_mean = getattr(self, f'sq_mean_{i}')
            space += mean.numel() * 4  # 4 bytes per float32
            space += sq_mean.numel() * 4
        return space