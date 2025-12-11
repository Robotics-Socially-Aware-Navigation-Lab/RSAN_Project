"""
Robust & scalable indoor scene classification pipeline.
-------------------------------------------------------

This script runs ONLY the scene classification portion of RSAN.
It does NOT perform object detection, symbolic reasoning, or LLM reasoning.

Primary uses:
    ✔ Classify a single image
    ✔ Classify all images in a folder
    ✔ Classify a video frame-by-frame
    ✔ Classify webcam stream in real-time
    ✔ Save annotated outputs to disk
    ✔ Store structured logs in JSON for reproducibility

Uses the RSAN IndoorClassifier (ResNet50 backbone + Places365 + MIT head).
This ensures identical behavior to the unified perception pipeline.

Outputs go to:
    outputs/classification/images/
    outputs/classification/videos/
    outputs/classification/results.json

Author: Rolando – RSAN Project
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2

# ------------------------------------------------------------
# Import the IndoorClassifier (MIT + Places365)
# ------------------------------------------------------------
# NOTE:
# This classifier produces:
#   - final_label (hybrid MIT+Places365)
#   - confidence
#   - per-class MIT probabilities
#
# It is the SAME classifier used inside unified_pipeline.py,
# ensuring consistent scene predictions everywhere in the RSAN Project.
from src.reasoning.indoor_classifier import IndoorClassifier

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logger = logging.getLogger("SceneClassifier")
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# Output directory setup
# ------------------------------------------------------------
OUTPUT_ROOT = Path("outputs/classification")
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
VIDEO_OUT_DIR = OUTPUT_ROOT / "videos"
RESULTS_JSON = OUTPUT_ROOT / "results.json"

IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Supported file extensions
# ------------------------------------------------------------
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPPORTED_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def timestamp() -> str:
    """Return a filesystem-safe timestamp for output naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path) -> bool:
    """Return True if the file extension is in SUPPORTED_IMG."""
    return p.suffix.lower() in SUPPORTED_IMG


def is_video(p: Path) -> bool:
    """Return True if the file extension is in SUPPORTED_VIDEO."""
    return p.suffix.lower() in SUPPORTED_VIDEO


def annotate_image(img, label: str, conf: float):
    """
    Draw the predicted scene label + confidence onto the image.

    This is a simplified version of the HUD from unified_pipeline.py.
    It only shows the scene label instead of full reasoning context.
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
    Append a structured result entry to results.json.

    Why:
        - Provides reproducible logs for every run.
        - Useful for debugging, experiments, dataset creation.
        - Mirrors unified_pipeline JSON logging.
    """
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# IMAGE CLASSIFICATION
# ============================================================


def classify_image(path: Path, clf: IndoorClassifier):
    """
    Classify a single image using the IndoorClassifier.

    Steps:
        1. Load image with OpenCV
        2. Predict scene label and confidence
        3. Draw annotation
        4. Save annotated image
        5. Log structured results to JSON
    """
    logger.info(f"[IMAGE] Classifying: {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Cannot read image: {path}")
        return

    # Run classifier (Places365 + MIT)
    result = clf.predict(img)

    # Create annotated output
    annotated = annotate_image(img.copy(), result.label, result.confidence)
    out_img_path = IMAGE_OUT_DIR / f"{path.stem}_classified_{timestamp()}.jpg"
    cv2.imwrite(str(out_img_path), annotated)

    # Store full result record
    record = {
        "type": "image",
        "input": str(path.resolve()),
        "output": str(out_img_path.resolve()),
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probs,  # MIT head probabilities
        "timestamp": timestamp(),
    }
    save_result_record(record)

    logger.info(f"[SAVED] → {out_img_path}")


# ============================================================
# VIDEO CLASSIFICATION
# ============================================================


def classify_video(path: Path, clf: IndoorClassifier):
    """
    Classify every frame of a video.

    Steps per frame:
        1. Read frame
        2. Predict scene label
        3. Annotate frame
        4. Write to output video

    After finishing:
        - Save frame-by-frame predictions to JSON
    """
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

        # Indoor scene prediction
        result = clf.predict(frame)

        # Save annotated frame
        annotated = annotate_image(frame.copy(), result.label, result.confidence)
        out_vid.write(annotated)

        # Store per-frame data
        frame_results.append({"frame": frame_idx, "label": result.label, "confidence": result.confidence})

        frame_idx += 1

    cap.release()
    out_vid.release()

    # Store JSON entry
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


# ============================================================
# FOLDER CLASSIFICATION
# ============================================================


def classify_folder(folder: Path, clf: IndoorClassifier):
    """
    Classify ALL images inside a folder.

    This does NOT recursively search subfolders.

    Used for:
        - Batch dataset labeling
        - Debugging classifier behavior on curated image sets
    """
    logger.info(f"[FOLDER] Classifying all images in: {folder}")

    images = sorted([p for p in folder.iterdir() if is_image(p)])
    if not images:
        logger.warning("No images found in folder.")
        return

    for img_path in images:
        classify_image(img_path, clf)


# ============================================================
# WEBCAM CLASSIFICATION
# ============================================================


def classify_webcam(clf: IndoorClassifier):
    """
    Real-time webcam scene classification.

    Press Q to exit.

    Useful for:
        - Live demos
        - Robot-onboard camera testing
        - Scene classifier validation
    """
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


# ============================================================
# MAIN ROUTER
# ============================================================


def run_scene_classification(input_path: str):
    """
    Unified router that directs input to the correct classification method.

    Accepts:
        - image path
        - video path
        - folder path
        - "webcam"
    """
    clf = IndoorClassifier()
    path = Path(input_path)

    if input_path.lower() == "webcam":
        return classify_webcam(clf)

    if path.is_file() and is_image(path):
        return classify_image(path, clf)

    if path.is_file() and is_video(path):
        return classify_video(path, clf)

    if path.is_dir():
        return classify_folder(path, clf)

    logger.error(f"Invalid path or unsupported file type: {input_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Command-line execution:
        python -m src.tools.run_scene_classification path/to/image.jpg
        python -m src.tools.run_scene_classification path/to/video.mp4
        python -m src.tools.run_scene_classification path/to/folder
        python -m src.tools.run_scene_classification webcam
    """
    import sys

    if len(sys.argv) > 1:
        run_scene_classification(sys.argv[1])
    else:
        print("Usage:")
        print("  python -m src.tools.run_scene_classification path/to/image.jpg")
        print("  python -m src.tools.run_scene_classification path/to/video.mp4")
        print("  python -m src.tools.run_scene_classification path/to/folder")
        print("  python -m src.tools.run_scene_classification webcam")
