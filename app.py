from flask import Flask, render_template, request, send_from_directory
import os
import torch
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv5 models correctly
from ultralytics import YOLO  # Import YOLO model class

# Load YOLOv5 models manually
# Load YOLOv5 and YOLOv8 models
models = {
    "YOLOv5s": YOLO(os.path.join(MODEL_FOLDER, "bestS.pt")),
    "YOLOv5m": YOLO(os.path.join(MODEL_FOLDER, "bestM.pt")),
    "YOLOv5l": YOLO(os.path.join(MODEL_FOLDER, "bestL.pt")),
    "YOLOv8n": YOLO(os.path.join(MODEL_FOLDER, "best8N.pt")),  # New YOLOv8n model
    "YOLOv8s": YOLO(os.path.join(MODEL_FOLDER, "best8S.pt")),  # New YOLOv8s model
    "YOLOv8m": YOLO(os.path.join(MODEL_FOLDER, "best8M.pt")),  # New YOLOv8m model
    "YOLOv8l": YOLO(os.path.join(MODEL_FOLDER, "best8L.pt")),  # New YOLOv8l model
    # "YOLOv8x": YOLO(os.path.join(MODEL_FOLDER, "best8X.pt"))   # New YOLOv8x model
}



# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to run inference
def run_inference(model, img_path, model_name):
    img = cv2.imread(img_path)  # Load the image using OpenCV
    results = model(img)  # Run inference using the loaded YOLOv5 model

    # Extract the bounding boxes, class IDs, and confidence scores
    boxes = results[0].boxes  # Get the bounding boxes
    class_ids = boxes.cls.cpu().numpy()  # Class IDs (as numpy array)
    confidences = boxes.conf.cpu().numpy()  # Confidence scores
    names = results[0].names  # Class names

    # Loop through the results and draw bounding boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()  # Get bounding box coordinates
        class_id = int(class_ids[i])  # Get the class ID
        label = names[class_id]  # Get the class name
        confidence = confidences[i]  # Confidence score

        # Draw bounding box and label on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed image
    output_filename = f"output_{model_name}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, img)

    return output_filename

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return "Invalid file"

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Run inference on all models
        output_files = {name: run_inference(model, filepath, name) for name, model in models.items()}

        return render_template("result.html", input_image=filename, output_files=output_files)

    return render_template("index.html")

@app.route("/static/outputs/<filename>")
def send_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
