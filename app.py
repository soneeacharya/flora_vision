from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
from model import CustomCNN

app = Flask(__name__)

# ----------------- Model Setup -----------------
model = CustomCNN(num_classes=12, input_size=(128, 128))
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

# Class labels (same as training)
classes = ['amala', 'baganbilas', 'daisy', 'dandelion', 'gulbahar', 'japakusum',
           'neem', 'rhododendron', 'rose', 'sunflower', 'tulip', 'tulsi']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No image selected")

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    prediction = classes[predicted.item()]
    confidence = confidence.item() * 100

    return render_template('result.html', image_path=image_path, prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
