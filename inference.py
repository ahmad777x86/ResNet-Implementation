import torch
from model import ResNet
from DataLoading import load_data
from utils import preprocess_image
import matplotlib.pyplot as plt

model = ResNet(in_channels=3, out_channels=64, classes=10)

model.load_state_dict(torch.load('model.pth'))

x = torch.randn(4,3,32,32)

model.eval()

with torch.inference_mode():
    y = model(x)

print(f"Y shape: {y.shape}")

raw_img = plt.imread('Test Images/frog1.jpg')

images = preprocess_image(raw_img)

images = images.unsqueeze(0)

with torch.inference_mode():
    logits = model(images)
    preds = torch.argmax(logits, dim=1)


print(f"Prediction: {preds[0]}")
plt.imshow(raw_img)
plt.show()