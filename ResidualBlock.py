import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        if(in_channels != out_channels or stride != 1):
            self.projection = nn.Conv2d(in_channels, out_channels, stride=stride, padding=1, bias=False)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        if identity is not None:
            identity = self.projection(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity

        x = self.bn2(x)

        return self.relu(x)