"""
Unified RSAN Perception Pipeline (Images, Videos, Webcam)
--------------------------------------------------------

Runs:
  • YOLO object detection
  • Indoor scene classification
  • Scene reasoning
  • Saves annotated outputs for:
        - Images
        - Videos
        - Webcam streams

Output directories:
    outputs/full_pipeline/images/
    outputs/full_pipeline/videos/
    outputs/full_pipeline/logs/

Author: Senior ML & Robotics Engineer
"""

import cv2
import json
import logging
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from src.reasoning.indoor_classifier import IndoorClassifier
from src.reasoning.scene_context import reason_about_scene

# ---------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------

ROOT = Path("outputs/full_pipeline")
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
# Helpers
# ---------------------------------------------------------

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def is_video(p: Path):
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def draw_predictions(frame, objects, scene_label, conf, reasoning):
    """
    Draw all annotations on the frame.
    """
    # Draw YOLO boxes
    for box, cls in zip(objects.boxes.xyxy, objects.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = objects.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Scene classification heading
    cv2.putText(frame,
                f"{scene_label.upper()} ({conf:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3)

    # Reasoning
    cv2.putText(frame,
                f"Crowd: {reasoning.crowd_level} | Risk: {reasoning.risk_score:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 200),
                2)

    cv2.putText(frame,
                f"Hint: {reasoning.navigation_hint}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 50, 255),
                2)

    return frame


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

def append_json(entry):
    if JSON_LOG.exists():
        with open(JSON_LOG, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(JSON_LOG, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

def load_models():
    detector = YOLO("models/yolo_detector/best.pt")
    classifier = IndoorClassifier()
    return detector, classifier


# ---------------------------------------------------------
# PROCESS IMAGE
# ---------------------------------------------------------

def process_image(path: Path, detector, classifier):
    logger.info(f"Processing image → {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Could not read: {path}")
        return

    det_res = detector(img)[0]
    cls_res = classifier.predict(img)
    reasoning = reason_about_scene(cls_res.label, det_res)

    img = draw_predictions(img, det_res, cls_res.label, cls_res.confidence, reasoning)

    out_path = IMG_OUT / f"{path.stem}_full_{timestamp()}.jpg"
    cv2.imwrite(str(out_path), img)

    logger.info(f"[IMAGE SAVED] → {out_path.resolve()}")

    append_json({
        "type": "image",
        "input": str(path.resolve()),
        "output": str(out_path.resolve()),
        "scene": cls_res.label,
        "confidence": cls_res.confidence,
        "timestamp": timestamp(),
    })


# ---------------------------------------------------------
# PROCESS VIDEO
# ---------------------------------------------------------

def process_video(path: Path, detector, classifier):
    logger.info(f"Processing video → {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {path}")
        return

    out_path = VID_OUT / f"{path.stem}_full_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_res = detector(frame)[0]
        cls_res = classifier.predict(frame)
        reasoning = reason_about_scene(cls_res.label, det_res)

        frame = draw_predictions(frame, det_res,
                                 cls_res.label, cls_res.confidence,
                                 reasoning)

        writer.write(frame)

    cap.release()
    writer.release()

    logger.info(f"[VIDEO SAVED] → {out_path.resolve()}")


# ---------------------------------------------------------
# PROCESS WEBCAM
# ---------------------------------------------------------

def process_webcam(index: int, detector, classifier):
    logger.info(f"Webcam started (device={index}) — Press Ctrl+C to stop")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        return

    out_path = VID_OUT / f"webcam_full_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            det_res = detector(frame)[0]
            cls_res = classifier.predict(frame)
            reasoning = reason_about_scene(cls_res.label, det_res)

            frame = draw_predictions(frame, det_res,
                                     cls_res.label, cls_res.confidence,
                                     reasoning)

            writer.write(frame)

    except KeyboardInterrupt:
        logger.info("Webcam stopped.")

    cap.release()
    writer.release()

    logger.info(f"[WEBCAM SAVED] → {out_path.resolve()}")


# ---------------------------------------------------------
# MASTER ROUTER
# ---------------------------------------------------------

def run_unified(input_path: str):
    detector, classifier = load_models()
    p = Path(input_path)

    # Webcam mode
    if input_path.lower() in {"webcam", "cam"}:
        return process_webcam(0, detector, classifier)

    # Single image
    if p.is_file() and is_image(p):
        return process_image(p, detector, classifier)

    # Single video
    if p.is_file() and is_video(p):
        return process_video(p, detector, classifier)

    # Folder (batch of images/videos)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if is_image(f):
                process_image(f, detector, classifier)
            elif is_video(f):
                process_video(f, detector, classifier)
        return

    logger.error(f"Invalid input: {input_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_unified(sys.argv[1])
    else:
        print("Usage:")
        print("   python -m src.tools.run_unified_pipeline <image|video|folder|webcam>")