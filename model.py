import os

# Path to your dataset inside Google Drive
DATASET_PATH = "/content/drive/MyDrive/YoloV8Final/data"  # Change this to your dataset folder path

# Images and annotations path
TRAIN_IMG_PATH = os.path.join(DATASET_PATH, "/images/train")
VAL_IMG_PATH = os.path.join(DATASET_PATH, "/images/val")
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "/labels/train")  # Your annotations
VAL_LABELS_PATH = os.path.join(DATASET_PATH, "/labels/val")
yaml_content = f"""
path: {DATASET_PATH}  # Root directory of dataset
train: images/train  # Training images
val: images/val  # Validation images

nc: 4  # Number of classes
names: ['people', 'ammunitions', 'suspect', 'vehicle']  # Class names
"""

with open("custom_config.yaml", "w") as f:
    f.write(yaml_content)

print("✅ Custom dataset YAML file created!")

from ultralytics import YOLO

# List of YOLOv5 model versions to download
yolov5_versions = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']

# Loop through each version and download the pretrained weights
for version in yolov5_versions:
    print(f"Downloading {version}...")

    # Download the model weights
    model = YOLO(version)  # This will automatically download the model weights if not available
    print(f"✅ {version} model downloaded!")
from ultralytics import YOLO

# List of YOLOv5 model versions to train
yolov5_versions = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
gdrive_path = "/content/drive/MyDrive/yolov5"
# Loop through each YOLOv5 version and train the model
for version in yolov5_versions:
    print(f"Training {version}...")

    # Load the corresponding YOLOv5 model
    model = YOLO(version)  # Load pretrained YOLOv5 model

    # Train the model
    results = model.train(
        data="custom_config.yaml",  # Path to your dataset YAML file
        epochs=50,  # Number of training epochs
        batch=16,  # Batch size
        imgsz=640,  # Image size
        workers=4,  # Number of CPU workers
        device="cuda"  # Use GPU if available
    )

    # Save the trained model to Google Drive
    model_path = os.path.join(gdrive_path, f"yolov5_{version.split('.')[0]}")
    model.save(model_path)  # Save the model to the specified path
    print(f"✅ {version} model saved to {model_path}")
import shutil

# Specify the paths of your trained models (if they are already saved in Colab)
trained_model_paths = [
    '/content/yolov5su.pt',
    '/content/yolov5mu.pt',
    '/content/yolov5lu.pt',
    '/content/yolov5xu.pt'
]

# Destination folder in Google Drive
gdrive_path = "/content/drive/MyDrive/yoloV5"

# Move models to Google Drive
for model_path in trained_model_paths:
    shutil.move(model_path, gdrive_path)
    print(f"Moved {model_path} to {gdrive_path}")