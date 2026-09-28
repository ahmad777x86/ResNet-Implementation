import gradio as gr
import spaces
import torch
from utils import preprocess_image
from model import ResNet


print("Initializing app...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print("Loading Model weights and preparing compilation...")

model = ResNet(3,64,10).to(device)

state_dict = torch.load('model.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to('cuda')


CLASSES = ["Airplane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

@spaces.GPU
def predict(img):
    img = preprocess_image(img, device)
    img = img.unsqueeze(0)

    logits = model(img)
    probabilities = torch.softmax(logits, dim=1)
    
    confidence, preds = torch.topk(probabilities, k=1, dim=1)
    conf = float(confidence[0]) * 100
    conf_str = f"{conf:.2f}%"
    class_idx = int(preds[0])

    return {"Class" : CLASSES[class_idx], "Confidence" : conf_str}

app = gr.Interface(predict, inputs=gr.Image(type='pil'), outputs=gr.JSON())

app.launch(inbrowser=True, ssr_mode=False)