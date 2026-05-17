<div align="center">

# Camouflage Detector: Military Objectss Camouflage Detection Web App

</div>

A full-stack AI application built with **Flask**, **OpenCV**, and **YOLOv5/YOLOv8**.  
Camofledge enables users to upload images and detect camouflaged objects such as people, weapons, vehicles, or suspects using multiple state-of-the-art pre-trained models.  
The platform is designed for **rapid prototyping**, **easy deployment**, and a **responsive user experience**.

---

## 🎯 Performance Highlights

- **Full-Stack AI Integration**: Flask web frontend with backend model inference for seamless user interaction.
- **Pre-trained Model Deployment**: YOLOv5 and YOLOv8 models for fast, accurate detection—no retraining required.
- **Image Upload & Visualization**: Upload images and get detection results with bounding boxes and confidence scores.
- **Rapid Prototyping**: Showcase real-world AI applications without lengthy training cycles.

---

## 🛠️ Technical Stack

**Backend**

- Flask – lightweight web framework for routing and templating
- OpenCV – image processing and annotation
- Ultralytics YOLO – pre-trained object detection models (YOLOv5, YOLOv8)
- Torch – model inference engine

**Frontend**

- HTML/CSS – responsive upload and results page

**Development Tools**

- Git & GitHub – version control and collaboration
- pip – Python package management

---

## 🚀 Features

- 📤 **Image Upload**: Upload images directly from your browser.
- 🤖 **Multi-Model Detection**: Run inference with YOLO variants (n, s, m, l, x).
- 🖼 **Result Visualization**: Get annotated outputs with detected objects and confidence scores.
- ☁ **Easy Deployment**: Ready to run locally or deploy on cloud platforms.

---

## 📦 Getting Started

### Cloning the Repository

```bash
git clone https://github.com/your-username/camofledge.git
```

### Install Requirements

Download the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

## 📸 App Preview

<table width="100%"> 
<tr>
<td width="50%">      
<p align="center">
  Home & Upload Image Page
</p>
<img src="https://github.com/Varshanth2025/military-camouflaged-object-detection/blob/main/static/images/home%20and%20image%20uploading%20page.png">
</td> 
<td width="50%">
<p align="center">
  Detection Page
</p>
<img src="https://github.com/Varshanth2025/military-camouflaged-object-detection/blob/main/static/images/detection.png">  
</td>
</tr>
</table>
