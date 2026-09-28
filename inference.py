import torch
import torch.profiler as profiler
import matplotlib.pyplot as plt
from model import ResNet
from DataLoading import load_data
from utils import preprocess_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
model = ResNet(in_channels=3, out_channels=64, classes=10).to(device)

model.load_state_dict(torch.load('model.pth'))

compiled_model = torch.compile(model)

compiled_model.eval()

raw_img = plt.imread('Test Images/horse1.jpg')

images = preprocess_image(raw_img, device)

images = images.unsqueeze(0)

# first pass for lazy compiling
with torch.inference_mode():
    _ = compiled_model(images)

torch.cuda.synchronize()

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
    profile_memory = True, 
    record_shapes = True
) as prof:
    with torch.inference_mode():
        logits = compiled_model(images)
        preds = torch.argmax(logits, dim=1)

prof.export_chrome_trace("./Profiles/cuda_trace2")

print(f"Prediction: {preds[0]}")
plt.imshow(raw_img)
plt.show()