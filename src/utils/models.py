from random import random
import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, random, numpy as np

def set_seed(seed: int = 111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

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


class LeNetVB(nn.Module):
    def __init__(self, num_classes=10, var0=1.0, estimator='flipout'):
        super(LeNetVB, self).__init__()

        from bayesian_torch.layers import Conv2dReparameterization, Conv2dFlipout
        from bayesian_torch.layers import LinearReparameterization, LinearFlipout

        Conv2dVB = Conv2dFlipout
        LinearVB = LinearFlipout

        self.conv1 = Conv2dVB(1, 6, 5, prior_variance=var0)
        self.conv2 = Conv2dVB(6, 16, 5, prior_variance=var0)
        self.flatten = nn.Flatten()
        self.fc1 = LinearVB(256, 120, prior_variance=var0)
        self.fc2 = LinearVB(120, 84, prior_variance=var0)
        self.fc3 = LinearVB(84, num_classes, prior_variance=var0)

    def features(self, x, return_acts=False):
        kl_total = 0

        x, kl = self.conv1(x)
        kl_total += kl
        x = F.max_pool2d(F.relu(x), 2, 2)

        x, kl = self.conv2(x)
        kl_total += kl
        x = F.max_pool2d(F.relu(x), 2, 2)

        x = self.flatten(x)
        x, kl = self.fc1(x)
        kl_total += kl
        x = F.relu(x)

        x, kl = self.fc2(x)
        kl_total += kl
        x = F.relu(x)

        return x, kl_total

    def forward(self, x):
        x, kl_total = self.features(x)
        x, kl = self.fc3(x)
        kl_total += kl
        return x, kl_total


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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


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