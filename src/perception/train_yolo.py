"""
train_yolo.py
Purpose:
    Train or fine-tune a YOLOv8 model on your prepared dataset.

How:
    - Loads dataset paths and configurations dynamically
    - Uses Ultralytics YOLO API
    - Saves model checkpoints and logs in designated folders

Why:
    - Automates model training and ensures reproducibility across team members
"""

from pathlib import Path

from ultralytics import YOLO

from utils.file_utils import load_paths
from utils.logger import setup_logger


def train_yolo():
    paths = load_paths()
    logger = setup_logger(name="train_yolo")

    data_yaml = Path(paths["yolo_ready"]) / "data.yaml"
    model_dir = paths["models"]

    logger.info("🚀 Starting YOLOv9 fine-tuning...")
    model = YOLO("yolov9.pt")  # Base model

    model.train(
        data=str(data_yaml), epochs=50, imgsz=640, project=str(model_dir), name="yolov9_finetuned", exist_ok=True
    )

    logger.info("YOLOv9 fine-tuning complete. Model saved in /models/")


if __name__ == "__main__":
    train_yolo()
