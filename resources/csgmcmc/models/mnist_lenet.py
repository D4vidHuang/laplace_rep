'''LeNet for MNIST in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class MNISTLeNet(nn.Module):
    def __init__(self):
        super(MNISTLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 channel input for MNIST
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)  # 4x4 feature maps for 28x28 input
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out 