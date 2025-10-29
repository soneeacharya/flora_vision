import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
import os

# import your CNN architecture
from plant_classifier.model import CustomCNN  # ensure same class name as in model.py

device = torch.device("cpu")

# Load model and weights
model = CustomCNN(num_classes=12)  # update number of classes
checkpoint = torch.load("E:/8semproject/best_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Class labels (same order used during training)
classes = ['amala', 'baganbilas', 'daisy', 'dandelion', 'gulbahar', 'japakusum',
           'neem', 'rhododendron', 'rose', 'sunflower', 'tulip', 'tulsi']

def index(request):
    result = None
    uploaded_image_url = None

    if request.method == 'POST' and 'image' in request.FILES:
        image = request.FILES['image']
        upload_dir = os.path.join('classifier', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, image.name)

        # Save uploaded image
        with open(file_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Predict
        img = Image.open(file_path)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = classes[predicted.item()]
            uploaded_image_url = f"/media/{image.name}"

    return render(request, 'classifier/index.html', {
        'result': result,
        'uploaded_image_url': uploaded_image_url
    })
