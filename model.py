import torch
import torch.nn as nn
from ResidualBlock import ResidualBlock

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, classes):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.block1 = ResidualBlock(out_channels, out_channels, stride=1)
        self.block2 = ResidualBlock(out_channels, 2*out_channels, stride=2)
        self.block3 = ResidualBlock(2*out_channels, 4*out_channels, stride=2)
        self.block4 = ResidualBlock(4*out_channels, 8*out_channels, stride=2)

        self.avg_pooling = nn.AvgPool2d(1,1)
        self.flatten = nn.Flatten(batch_size, 512)
        self.linear = nn.Linear(512, classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avg_pooling(x)
        x = self.flatten(x)
        return self.linear(x)
