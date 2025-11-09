"""
detect_objects.py
Purpose:
    Run object detection on new images/videos using a fine-tuned YOLO model.

How:
    - Loads YOLOv9_finetuned.pt from /models/
    - Runs inference on a target folder (default: /datasets/yolo_ready/images/val)
    - Saves detection results (images + JSON + TXT) in /results/detections/

Why:
    - Validates model accuracy visually and quantitatively
    - Provides detection data for reasoning and evaluation
"""

from utils.file_utils import load_paths, ensure_dirs
from utils.logger import setup_logger
from ultralytics import YOLO
from pathlib import Path

def run_detection(input_folder="datasets/yolo_ready/images/val"):
    paths = load_paths()
    ensure_dirs(paths)
    logger = setup_logger(name="detect_objects")

    model_path = Path(paths["models"]) / "yolov9_finetuned.pt"
    output_dir = Path(paths["results"]) / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running object detection...")
    model = YOLO(str(model_path))

    model.predict(
        source=input_folder,
        save=True,
        save_txt=True,
        project=str(output_dir),
        name="detections",
        exist_ok=True
    )

    logger.info(f"Detections saved in {output_dir}")

if __name__ == "__main__":
    run_detection()