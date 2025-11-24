"""
Robust & scalable indoor scene classification pipeline.
-------------------------------------------------------

Features:
✔ Classify single images
✔ Classify entire folders
✔ Save annotated output images
✔ Log all results to JSON for reproducibility
✔ MacOS-safe (no OpenCV GUI windows)
✔ Uses RSAN IndoorClassifier model
✔ Plug-and-play with your project structure

Outputs:
    outputs/classification/images/
    outputs/classification/results.json

Author: Senior Machine Learning Engineer – RSAN Project
"""

import cv2
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.reasoning.indoor_classifier import IndoorClassifier

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("SceneClassifier")
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Output directories
# -------------------------------------------------------------------
OUTPUT_ROOT = Path("outputs/classification")
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
RESULTS_JSON = OUTPUT_ROOT / "results.json"

IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def annotate_image(img, label: str, conf: float):
    """
    Draw label + confidence on the image.
    """
    text = f"{label.upper()} ({conf:.2f})"
    cv2.putText(
        img,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def save_result_record(record: Dict[str, Any]):
    """
    Append results to JSON log (robust, scalable).
    """
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=4)


# -------------------------------------------------------------------
# Core classification logic
# -------------------------------------------------------------------
def classify_image(path: Path, clf: IndoorClassifier):
    """
    Runs classification on a single image.
    Saves annotated image and logs results in JSON.
    """

    logger.info(f"[CLASSIFY] {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Cannot read image: {path}")
        return

    result = clf.predict(img)

    # Save annotated image
    annotated = annotate_image(img.copy(), result.label, result.confidence)
    out_img_path = IMAGE_OUT_DIR / f"{path.stem}_classified_{timestamp()}.jpg"
    cv2.imwrite(str(out_img_path), annotated)

    # Save structured record
    record = {
        "input_image": str(path.resolve()),
        "output_image": str(out_img_path.resolve()),
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probs,
        "timestamp": timestamp(),
    }
    save_result_record(record)

    logger.info(f"[SAVED] Annotated → {out_img_path}")
    logger.info(f"[LOGGED] Entry → {RESULTS_JSON}")


def classify_folder(folder: Path, clf: IndoorClassifier):
    """
    Classify every image in a directory.
    """
    logger.info(f"[FOLDER] Classifying all images inside: {folder}")

    images = sorted([p for p in folder.iterdir() if is_image(p)])
    if not images:
        logger.warning("No images found in folder.")
        return

    for img_path in images:
        classify_image(img_path, clf)

    logger.info("[DONE] Folder classification completed.")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def run_scene_classification(input_path: str):
    """
    Main router for all classification modes.
    """

    path = Path(input_path)
    clf = IndoorClassifier()  # loads your model automatically

    if path.is_file() and is_image(path):
        return classify_image(path, clf)

    if path.is_dir():
        return classify_folder(path, clf)

    logger.error(f"Invalid path or unsupported file type: {input_path}")


# -------------------------------------------------------------------
# Command-line execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_scene_classification(sys.argv[1])
    else:
        print("Usage:")
        print("  python -m src.tools.run_scene_classification path/to/image.jpg")
        print("  python -m src.tools.run_scene_classification path/to/folder")
