import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

print("Initializing app...")

session = ort.InferenceSession(
    'model.onnx',
    providers=['CPUExecutionProvider']
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded successfully!")

def preprocess_for_onnx(image_input):

    if isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError("Input must be a NumPy array or PIL Image")

    w, h = img.size
    if h < w:
        new_h, new_w = 32, int(w * 32 / h)
    else:
        new_h, new_w = int(h * 32 / w), 32
    
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    w, h = img.size
    left = (w - 32) // 2
    top = (h - 32) // 2
    right = left + 32
    bottom = top + 32
    img = img.crop((left, top, right, bottom))

    arr = np.array(img, dtype=np.float32) / 255.0
    
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
        
    arr = np.transpose(arr, (2, 0, 1)) 
    arr = np.expand_dims(arr, axis=0)

    return arr

CLASSES = ["Airplane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def predict(img):
    img = preprocess_for_onnx(img)

    output = session.run(
        {"output"},
        {input_name: img}
    )[0]
    logits = output[0]
    norm_logits = logits - np.max(logits)
    probabilities = np.exp(norm_logits)
    probabilities = probabilities/probabilities.sum()
    class_idx = int(np.argmax(probabilities))

    conf = float(probabilities[class_idx]*100)
    conf_str = f"{conf:.2f}%"

    return {"Class" : CLASSES[class_idx], "Confidence" : conf_str}

app = gr.Interface(
    predict, 
    inputs=gr.Image(type='pil'), 
    outputs=gr.JSON(),
    title="Resnet-18 CIFAR Classifier", 
    description="#### Description: Trained on cifar-10 and may classify images of Airplanes, Cars, Trucks, Ship, Cats, Dogs, Deer, Frogs, Horse, Birds.",
    article= "Simply choose one of the above images or upload your own image and click submit, and the output prediction shall appear!",
    examples=['Test Images/airplane1.jpg', 'Test Images/car1.jpg', 'Test Images/deer1.jpg']
)

app.launch(
    inbrowser=True, 
    ssr_mode=False,
    theme=gr.themes.Soft()
)

