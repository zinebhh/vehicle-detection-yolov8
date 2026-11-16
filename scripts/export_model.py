"""Export du modèle pour le déploiement (ONNX / TensorRT / OpenVINO...).
Exemple : python scripts/export_model.py --weights runs/yolov8_vehicles/weights/best.pt --format onnx
"""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/yolov8_vehicles/weights/best.pt")
p.add_argument("--format", default="onnx", help="onnx, engine, openvino, torchscript, tflite")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--half", action="store_true", help="FP16 (GPU)")
a = p.parse_args()
print("Export :", YOLO(a.weights).export(format=a.format, imgsz=a.imgsz, half=a.half))
