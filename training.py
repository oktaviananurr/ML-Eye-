
# ======================================
# 1. Install dan Import Library
# ======================================
import os
os.system("pip install roboflow ultralytics --upgrade")

from roboflow import Roboflow
from ultralytics import YOLO

# ======================================
# 2. Download Dataset dari Roboflow (format: YOLOv8)
# ======================================
rf = Roboflow(api_key="1cAnwA5hzVayU4ikxPJQ")

# GANTI di sini: workspace, project ID, dan versi sesuai dataset Anda

project = rf.workspace("hifivetech").project("eye_diseases_detect")
version = project.version(1)

# Download ke format YOLOv8
dataset = version.download("yolov8")

# ======================================
# 3. Training Model YOLOv8
# ======================================
# Gunakan model YOLOv8s (small), bisa juga yolov8m atau yolov8n
model = YOLO("yolov8s.pt")  # Gunakan pretrained weight YOLOv8

# Training
model.train(
    data=f"eye_diseases_detect-1/data.yaml",  # path ke data.yaml
    epochs=100,
    imgsz=416,
    batch=16,
    name="YOLOv8_EYE_DISEASES",  # nama folder output
    project="runs/train",   # direktori output
    exist_ok=True
)

# ======================================
# 4. Simpan atau Load Model Trained
# ======================================
# Setelah training selesai, model tersimpan di:
# runs/train/YOLOv8_KULIT/weights/best.pt

# Load model jika ingin digunakan lagi:
# model = YOLO("runs/train/YOLOv8_KULIT/weights/best.pt")
  