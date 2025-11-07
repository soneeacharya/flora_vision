import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from model import CustomCNN
import os
from PIL import Image
import pillow_avif
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- PATHS -----------------
train_dir = "E:/8semproject/dataset_split/train"
val_dir = "E:/8semproject/dataset_split/val"

# ----------------- HYPERPARAMETERS -----------------
batch_size = 32
epochs = 50
learning_rate = 0.001
image_size = 128
patience = 5


# ----------------- DEVICE -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- TRANSFORMS -----------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------- SAFE LOADER -----------------
def safe_loader(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as e:
        print(f"⚠ Skipping corrupted image: {path} ({e})")
        return None

# ----------------- DATASETS -----------------
train_data = datasets.ImageFolder(train_dir, transform=transform, loader=safe_loader)
val_data = datasets.ImageFolder(val_dir, transform=transform, loader=safe_loader)

# Handle class imbalance with WeightedRandomSampler
targets = [label for _, label in train_data.samples]
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=batch_size)

classes = train_data.classes
print("Classes:", classes)

# ----------------- MODEL -----------------
model = CustomCNN(num_classes=len(classes), input_size=(image_size, image_size)).to(device)

# ----------------- OPTIMIZER & LOSS -----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ----------------- RESUME TRAINING IF CHECKPOINT EXISTS -----------------
start_epoch = 0
best_val_acc = 0.0
checkpoint_path = "checkpoint.pth"

if os.path.exists(checkpoint_path):
    print("🔄 Found checkpoint! Resuming training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_val_acc = checkpoint["best_val_acc"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch} with best val acc {best_val_acc:.2f}%")
else:
    print("🆕 Starting fresh training...")

# ----------------- TRAINING LOOP -----------------
patience_counter = 0
for epoch in range(start_epoch, epochs):
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

    # ----------------- VALIDATION -----------------
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

    # ----------------- SAVE CHECKPOINT -----------------
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc
    }, checkpoint_path)

    # ----------------- SAVE BEST MODEL -----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"💾 Saved new best model (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered — no improvement.")
            break

print("🎉 Training completed. Best Validation Accuracy:", best_val_acc)
