
"""
Robust & scalable indoor scene classification pipeline.
-------------------------------------------------------

Features:
✔ Classify single images
✔ Classify entire folders
✔ Classify videos frame-by-frame
✔ Classify webcam stream (press Q to quit)
✔ Save annotated output images or videos
✔ Log all results to JSON for reproducibility
✔ MacOS-safe (no OpenCV GUI popups unless webcam)
✔ Uses RSAN IndoorClassifier model
✔ Plug-and-play with the project structure

Outputs:
    outputs/classification/images/
    outputs/classification/videos/
    outputs/classification/results.json

Author: Rolando – RSAN Project
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
VIDEO_OUT_DIR = OUTPUT_ROOT / "videos"
RESULTS_JSON = OUTPUT_ROOT / "results.json"

IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPPORTED_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_IMG


def is_video(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_VIDEO


def annotate_image(img, label: str, conf: float):
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
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=4)


# -------------------------------------------------------------------
# Image Classification
# -------------------------------------------------------------------
def classify_image(path: Path, clf: IndoorClassifier):
    logger.info(f"[IMAGE] Classifying: {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Cannot read image: {path}")
        return

    result = clf.predict(img)

    annotated = annotate_image(img.copy(), result.label, result.confidence)
    out_img_path = IMAGE_OUT_DIR / f"{path.stem}_classified_{timestamp()}.jpg"
    cv2.imwrite(str(out_img_path), annotated)

    record = {
        "type": "image",
        "input": str(path.resolve()),
        "output": str(out_img_path.resolve()),
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probs,
        "timestamp": timestamp(),
    }
    save_result_record(record)

    logger.info(f"[SAVED] → {out_img_path}")


# -------------------------------------------------------------------
# Video Classification
# -------------------------------------------------------------------
def classify_video(path: Path, clf: IndoorClassifier):
    logger.info(f"[VIDEO] Classifying video: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(3))
    height = int(cap.get(4))

    out_path = VIDEO_OUT_DIR / f"{path.stem}_classified_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = clf.predict(frame)

        annotated = annotate_image(frame.copy(), result.label, result.confidence)
        out_vid.write(annotated)

        frame_results.append({
            "frame": frame_idx,
            "label": result.label,
            "confidence": result.confidence
        })

        frame_idx += 1

    cap.release()
    out_vid.release()

    record = {
        "type": "video",
        "input": str(path.resolve()),
        "output": str(out_path.resolve()),
        "frames": frame_idx,
        "frame_results": frame_results,
        "timestamp": timestamp(),
    }
    save_result_record(record)

    logger.info(f"[SAVED] → {out_path}")


# -------------------------------------------------------------------
# Folder Classification
# -------------------------------------------------------------------
def classify_folder(folder: Path, clf: IndoorClassifier):
    logger.info(f"[FOLDER] Classifying all images in: {folder}")

    images = sorted([p for p in folder.iterdir() if is_image(p)])
    if not images:
        logger.warning("No images found in folder.")
        return

    for img_path in images:
        classify_image(img_path, clf)


# -------------------------------------------------------------------
# Webcam Classification
# -------------------------------------------------------------------
# how webcam clasification works:
# Webcam → YOLO Detector → Indoor Classifier → Scene Reasoner → Rendered Output
# OpenCV (cv2) inside:
# It pass the webcam frame into the YOLOv8 detector:
# then it taeks the detected regions and feeds them into the IndoorClassifier:
# finally, it uses the SceneReasoner to interpret the classified scenes.

def classify_webcam(clf: IndoorClassifier):
    logger.info("[WEBCAM] Starting classification… (press Q to quit)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam not available.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = clf.predict(frame)
        annotated = annotate_image(frame.copy(), result.label, result.confidence)

        cv2.imshow("Scene Classification (Press Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------
# Main router
# -------------------------------------------------------------------
def run_scene_classification(input_path: str):
    clf = IndoorClassifier()
    path = Path(input_path)

    if input_path == "webcam":
        return classify_webcam(clf)

    if path.is_file() and is_image(path):
        return classify_image(path, clf)

    if path.is_file() and is_video(path):
        return classify_video(path, clf)

    if path.is_dir():
        return classify_folder(path, clf)

    logger.error(f"Invalid path or unsupported file type: {input_path}")


# -------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_scene_classification(sys.argv[1])
    else:
        print("Usage:")
        print("  python -m src.tools.run_scene_classification path/to/image.jpg")
        print("  python -m src.tools.run_scene_classification path/to/video.mp4")
        print("  python -m src.tools.run_scene_classification path/to/folder")
        print("  python -m src.tools.run_scene_classification webcam")