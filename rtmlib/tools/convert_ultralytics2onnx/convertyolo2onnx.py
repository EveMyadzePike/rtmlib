import os
import shutil
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
model.export(format="onnx", imgsz=[480,640])

onnx_path = "yolov8m.onnx"

# Define the target directory
target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..', '..', 'yolo_models')

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Move the ONNX file to the target directory
shutil.move(onnx_path, os.path.join(target_dir, onnx_path))

print(f"ONNX model moved to {target_dir}")