# AgingSignsDetection

# 🧠 Aging Signs Detection Using Deep Learning

## 📌 Overview
This project focuses on detecting visible signs of aging from facial images using a deep learning-based object detection pipeline. Built with **Python** and **TensorFlow 2.x**, it leverages the **EfficientDet** model to identify aging indicators such as wrinkles, eye bags, and other facial features.

The system includes both training and deployment pipelines, designed to work seamlessly in GPU environments like Google Colab. A simple user interface is provided using **Streamlit**, with **Ngrok** used for public web access.

---

## 🛠️ Features
- Custom object detection model for facial aging signs
- Full training pipeline using TensorFlow Object Detection API
- TFRecord generation for dataset preprocessing
- Streamlit-based inference app with Ngrok tunnel
- Google Drive integration for model access

---

## 🔍 Model Used
- **EfficientDet-D0**
  - Lightweight and real-time capable
  - Pre-trained on COCO dataset and fine-tuned for aging signs
  - Integrated via TensorFlow Object Detection API

---

## 🧪 Training Workflow
1. Prepare labeled image dataset
2. Generate TFRecord files using `generate_tfrecord.py`
3. Configure training pipeline in `model_main_tf2.py`
4. Train the model using EfficientDet base weights
5. Export the model with `exporter_main_v2.py`

---
