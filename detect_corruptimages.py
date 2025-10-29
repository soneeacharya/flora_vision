from PIL import Image
import os

# Replace with your dataset path
dataset_dirs = [
    "E:/8semproject/dataset_split/train",
    "E:/8semproject/dataset_split/val"
]

for dataset_dir in dataset_dirs:
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                # Try to open the image
                img = Image.open(path)
                img.verify()  # Check if image is okay
            except Exception:
                print(f"⚠ Removing corrupted image: {path}")
                os.remove(path)

