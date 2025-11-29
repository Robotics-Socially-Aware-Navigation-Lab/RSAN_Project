"""
Unified RSAN Perception Pipeline
--------------------------------

Runs:
  • YOLO object detection
  • Indoor scene classification
  • Scene reasoning
  • Saves annotated outputs for:
        - Images
        - Videos
        - Webcam streams

Output directories come from:
    configs/project_paths.yaml

Author: Rolando Yax
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils.file_utils import load_paths
from src.reasoning.indoor_classifier import IndoorClassifier
from src.reasoning.scene_context import reason_about_scene


# ---------------------------------------------------------
# LOAD PATHS FROM YAML
# ---------------------------------------------------------
paths = load_paths()

ROOT = paths["full_pipeline"]
IMG_OUT = ROOT / "images"
VID_OUT = ROOT / "videos"
LOG_DIR = ROOT / "logs"

IMG_OUT.mkdir(parents=True, exist_ok=True)
VID_OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

JSON_LOG = LOG_DIR / "results.json"

logger = logging.getLogger("unified_pipeline")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".png"}


def is_video(p: Path):
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def append_json(entry):
    """Append a new entry to results.json safely."""
    if JSON_LOG.exists():
        with open(JSON_LOG, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(JSON_LOG, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------
def draw_predictions(frame, objects, scene_label, conf, reasoning):
    """Draw YOLO, Scene Classification, and Reasoning onto the frame."""

    # Draw YOLO boxes
    for box, cls in zip(objects.boxes.xyxy, objects.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = objects.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Scene label
    cv2.putText(frame, f"{scene_label.upper()} ({conf:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    # Reasoning
    cv2.putText(
        frame,
        f"Crowd: {reasoning.crowd_level} | Risk: {reasoning.risk_score:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 200),
        2,
    )

    cv2.putText(frame, f"Hint: {reasoning.navigation_hint}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

    return frame


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
def load_models():
    detector = YOLO(str(paths["models"] / "yolo_detector" / "best.pt"))
    classifier = IndoorClassifier(device="cpu")
    return detector, classifier


# ---------------------------------------------------------
# Process image
# ---------------------------------------------------------
def process_image(path: Path, detector, classifier):
    logger.info(f"[IMAGE] {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Could not read image: {path}")
        return

    det = detector(img)[0]
    cls = classifier.predict(img)
    reasoning = reason_about_scene(cls.label, det)

    frame = draw_predictions(img, det, cls.label, cls.confidence, reasoning)

    out_path = IMG_OUT / f"{path.stem}_full_{timestamp()}.jpg"
    cv2.imwrite(str(out_path), frame)
    logger.info(f"Saved → {out_path}")

    append_json(
        {
            "type": "image",
            "input": str(path.resolve()),
            "output": str(out_path.resolve()),
            "scene": cls.label,
            "confidence": cls.confidence,
            "timestamp": timestamp(),
        }
    )


# ---------------------------------------------------------
# Process video
# ---------------------------------------------------------
def process_video(path: Path, detector, classifier):
    logger.info(f"[VIDEO] {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {path}")
        return

    out_path = VID_OUT / f"{path.stem}_full_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(3)), int(cap.get(4))

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detector(frame)[0]
        cls = classifier.predict(frame)
        reasoning = reason_about_scene(cls.label, det)

        frame = draw_predictions(frame, det, cls.label, cls.confidence, reasoning)
        writer.write(frame)

    cap.release()
    writer.release()

    logger.info(f"Saved video → {out_path}")


# ---------------------------------------------------------
# Process webcam (fixed for smooth playback)
# ---------------------------------------------------------
def process_webcam(detector, classifier, index: int = 0):
    logger.info("[WEBCAM] Starting webcam (Ctrl+C to stop)")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Webcam cannot be accessed.")
        return

    out_path = VID_OUT / f"webcam_full_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Force stable framerate
    fps = 30  # FIX: stable fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            det = detector(frame)[0]
            cls = classifier.predict(frame)
            reasoning = reason_about_scene(cls.label, det)

            frame = draw_predictions(frame, det, cls.label, cls.confidence, reasoning)

            writer.write(frame)

            # FIX: smooth 30 FPS playback
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord("q"):
                break

            cv2.imshow("Unified Pipeline Webcam", frame)

    except KeyboardInterrupt:
        logger.info("Webcam stopped")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    logger.info(f"Saved webcam → {out_path}")


# ---------------------------------------------------------
# Master router
# ---------------------------------------------------------
def run_unified(input_path: str):
    detector, classifier = load_models()
    p = Path(input_path)

    # Webcam
    if input_path.lower() in {"webcam", "cam"}:
        return process_webcam(detector, classifier)

    # Single image
    if p.is_file() and is_image(p):
        return process_image(p, detector, classifier)

    # Single video
    if p.is_file() and is_video(p):
        return process_video(p, detector, classifier)

    # Folder of files
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if is_image(f):
                process_image(f, detector, classifier)
            elif is_video(f):
                process_video(f, detector, classifier)
        return

    logger.error(f"Invalid input: {input_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_unified(sys.argv[1])
    else:
        print("Usage:")
        print("  python -m src.tools.run_unified_pipeline <image|video|folder|webcam>")
