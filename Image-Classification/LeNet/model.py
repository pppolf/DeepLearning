import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_channels=1):
        """
        in_chnnels: 通道数
        - MNIST 数据集通道数为 1
        - CIFAR10 数据集通道数为 3
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5) # (ch, 32, 32) -> (6, 28, 28)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # (6, 28, 28) -> (6, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # (6, 14, 14) -> (16, 10, 10)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # (16, 10, 10) -> (16, 5, 5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x