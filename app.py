import gradio as gr
import torch
from utils import preprocess_image
from model import ResNet

print("Initializing app...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print("Loading Model weights and preparing compilation...")

model = ResNet(3,64,10).to(device)

model.load_state_dict(torch.load('model.pth'))

original_model = model

try:
    model = torch.compile(model)
    with torch.inference_mode():
        _ = model(torch.randn(1, 3, 32, 32))
    print(f"Successfully initiated and compiled model")

except Exception as e:
    model = original_model
    print(f"Failed to compile model: {e}")

CLASSES = ["Airplane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def predict(img):
    img = preprocess_image(img, device)
    img = img.unsqueeze(0)

    logits = model(img)
    confidence, preds = torch.topk(logits, k=1, dim=1)
    conf = float(confidence[0])
    class_idx = int(preds[0])

    return {"Class" : CLASSES[class_idx], "Confidence" : {conf}}

app = gr.Interface(predict, inputs=gr.Image(type='pil'), outputs=gr.JSON())

app.launch(inbrowser=True, ssr_mode=False)

