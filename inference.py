import torch
from model import ResNet

model = ResNet(in_channels=3, out_channels=64, classes=10)

model.load_state_dict(torch.load('model.pth'))

x = torch.randn(4,3,32,32)

y = model(x)

print(f"Y shape: {y.shape}")