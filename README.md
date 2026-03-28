# StopAISlop
IRIS Short Term Project, an AI Generated Image Detector website with simple structure using Xception as a prediction model.
By: Erin Josephine Manik & Annisa Rizki

# StopAISlop: AI Image Detector
StopAISlop is a deep learning web application for detecting whether an image is **AI-generated** or **real**.  
It combines a **fine-tuned Xception model** with a clean web interface so users can upload an image, run inference, and view both the prediction result and supporting visual explanations.

## Overview
As generative AI becomes better at producing highly realistic images, distinguishing synthetic visuals from authentic ones becomes increasingly difficult. This project was built as an attempt to address that challenge by creating an accessible image verification tool.

Instead of stopping at model training in a notebook, this project brings the model into a full website experience. Users can:

- upload an image for analysis
- view the predicted class
- inspect confidence and class probabilities
- explore a simple explainability section
- review the model architecture and evaluation summary

## Features
- **AI vs Real image classification**
- **Flask backend** for inference
- **Xception-based model** for binary classification
- **Drag-and-drop image upload**
- **Support for JPG, PNG, and WEBP**
- **Confidence score and probability breakdown**
- **Explainability section with attention/heatmap view**
- **Model dashboard with architecture and evaluation metrics**
- **Dark mode toggle**
- **Sample image testing**

## Tech Stack

### Backend
- Python
- Flask
- PyTorch
- timm
- torchvision
- Pillow
- NumPy

### Frontend
- HTML
- CSS
- Vanilla JavaScript
- Lucide Icons

## Model Details

- **Backbone:** Xception
- **Framework:** PyTorch
- **Task:** Binary classification
- **Classes:** Real vs AI-Generated
- **Input preprocessing:** Resize to `299 × 299`, normalize using ImageNet mean and standard deviation
- **Inference output:** predicted label, confidence score, AI probability, real probability, and confidence level

## Project Structure

```bash
StopAISlop/
│
├── backend/
│   ├── app.py
│   └── model.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── models/
│   └── best_xception_tiny_genimage.pth
│
└── README.md
