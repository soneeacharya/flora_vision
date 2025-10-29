import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from plant_classifier.model import CustomCNN
from torchvision.datasets import ImageFolder

# ------------------ Custom Dataset to return paths ------------------
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)  # (image, label)
        path = self.imgs[index][0]  # image file path
        return original_tuple + (path,)

# ------------------ Settings ------------------
image_size = 128
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ Load checkpoint ------------------
checkpoint = torch.load("best_model.pth", map_location=device)
classes = checkpoint["classes"]
print("Classes:", classes)

# Rebuild model
model = CustomCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------------------ DataLoader for test set ------------------
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

test_dir = "E:/8semproject/dataset_split/test"  # adjust if needed
test_data = ImageFolderWithPaths(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ------------------ Evaluation ------------------
all_preds, all_labels, all_paths = [], [], []

with torch.no_grad():
    for images, labels, paths in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)

# ------------------ Save predictions ------------------
results = []
for path, true, pred in zip(all_paths, all_labels, all_preds):
    results.append({
        "image_path": path,
        "true_label": classes[true],
        "predicted_label": classes[pred]
    })

df = pd.DataFrame(results)

# 📌 Save predictions here
save_path = "E:/8semproject/predictions.csv"
df.to_csv(save_path, index=False)
print(f"✅ Predictions saved to {save_path}")

# ------------------ Reports ------------------
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=classes))
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
