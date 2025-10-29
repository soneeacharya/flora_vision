import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from plant_classifier.model import CustomCNN
import os
import pillow_avif


# Adjust paths
train_dir = "E:/8semproject/dataset_split/train"
val_dir = "E:/8semproject/dataset_split/val"

# Hyperparameters
batch_size = 32
epochs = 10
learning_rate = 0.001
image_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),           # flip images to generalize
    transforms.RandomRotation(15),               # rotate ±15 degrees
    transforms.ColorJitter(0.2, 0.2, 0.2),      # random brightness/contrast/saturation
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

from PIL import Image

def safe_loader(path):
    try:
        return Image.open(path).convert('RGB')  # force RGB
    except Exception as e:
        print(f"⚠ Skipping image {path}: {e}")
        return None

# Datasets
train_data = datasets.ImageFolder(train_dir, transform=transform, loader=safe_loader)
val_data = datasets.ImageFolder(val_dir, transform=transform, loader=safe_loader)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Classes
classes = train_data.classes
print("Classes:", classes)

# ----------------- Model -----------------
model = CustomCNN(num_classes=len(classes), input_size=(image_size, image_size)).to(device)
criterion = nn.CrossEntropyLoss()

# ----------------- Optimizer & Scheduler -----------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
# StepLR reduces LR by 0.5 every 5 epochs for smoother convergence

# ----------------- Training Loop with Early Stopping -----------------
best_val_acc = 0.0
patience, patience_counter = 5, 0  # stop if no improvement for 5 epochs

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    scheduler.step()

    # ----------------- Validation -----------------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Val Acc: {val_acc:.2f}%")

    # ----------------- Early Stopping & Model Saving -----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered. No improvement for", patience, "epochs.")
            break

print("Training completed. Best Validation Accuracy:", best_val_acc)