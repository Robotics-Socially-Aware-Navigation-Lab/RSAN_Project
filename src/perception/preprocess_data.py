# """
# Handles dataset cleaning, filtering, and augmentation using FiftyOne.
# """
# from utils.file_utils import load_paths, ensure_dirs
# from utils.logger import setup_logger
# from pathlib import Path
# import shutil

# def preprocess_dataset():
#     paths = load_paths()
#     ensure_dirs(paths)
#     logger = setup_logger(name="preprocess_data")

#     logger.info("Starting dataset preprocessing...")
#     raw_path = paths["raw_data"]
#     processed_path = paths["processed_data"]
#     yolo_path = paths["yolo_ready"]

#     # Placeholder logic: In real code, use FiftyOne or Albumentations for filtering
#     logger.info(f"Reading data from {raw_path}")
#     logger.info(f"Exporting YOLO-ready data to {yolo_path}")

#     # Example: copy processed data
#     for p in Path(raw_path).glob("*.jpg"):
#         shutil.copy(p, yolo_path / "images/train")

#     logger.info("Preprocessing complete.")

# if __name__ == "__main__":
#     preprocess_dataset()

"""
preprocess_data.py
Purpose:
    Cleans raw datasets (COCO, SUN RGB-D, etc.) and converts them into YOLO format.

How:
    - Loads directory paths using file_utils
    - Logs every preprocessing step
    - Copies or filters images into YOLO structure (train/val folders)
    - (Optional) can integrate FiftyOne or Albumentations for augmentation

Why:
    - To ensure clean, well-structured, and reproducible training data
"""

from utils.file_utils import load_paths, ensure_dirs
from utils.logger import setup_logger
from pathlib import Path
import shutil

def preprocess_dataset():
    paths = load_paths()
    ensure_dirs(paths)
    logger = setup_logger(name="preprocess_data")

    logger.info("🚀 Starting dataset preprocessing...")

    raw_path = paths["raw_data"]
    yolo_path = paths["yolo_ready"]

    # Here you could add FiftyOne filtering or image augmentation.
    # For now, we simulate data cleaning by copying raw files to YOLO-ready structure.
    for p in Path(raw_path).glob("*.jpg"):
        shutil.copy(p, yolo_path / "images/train")

    logger.info(f"Preprocessing complete. YOLO-ready data stored in: {yolo_path}")

if __name__ == "__main__":
    preprocess_dataset()