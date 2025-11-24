"""
Unified multimodal detection pipeline (OpenCV-safe, macOS-safe).
Images and videos are saved to disk instead of using cv2.imshow(),
which freezes on macOS/VS Code.

Outputs are stored in:
    outputs/detections/images/
    outputs/detections/videos/

Author: Senior ML Engineer / Robotics Navigation Specialist
"""

import cv2
import logging
from pathlib import Path
from datetime import datetime

from src.perception.detect_utils import run_detection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# GLOBAL OUTPUT ROOT
# ---------------------------------------------------------
OUTPUT_ROOT = Path("outputs/detections")
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
VIDEO_OUT_DIR = OUTPUT_ROOT / "videos"

# Ensure dirs exist
IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_video(p: Path):
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------
def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{det.class_name} {det.confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


# ---------------------------------------------------------
# Image processing (✔ Never freezes)
# ---------------------------------------------------------
def process_image(path: Path):
    logger.info(f"Processing image: {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Cannot read: {path}")
        return

    detections = run_detection(img)
    img = draw_boxes(img, detections)

    out_path = IMAGE_OUT_DIR / f"detected_{timestamp()}.jpg"
    cv2.imwrite(str(out_path), img)

    logger.info(f"[SAVED] {out_path.resolve()}")


# ---------------------------------------------------------
# VIDEO processing (✔ Never freezes)
# ---------------------------------------------------------
def process_video(path: Path):
    logger.info(f"Processing video: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return

    out_path = VIDEO_OUT_DIR / f"detected_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    logger.info(f"[WRITING] Processed video → {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_detection(frame)
        frame = draw_boxes(frame, detections)

        writer.write(frame)

    cap.release()
    writer.release()
    logger.info(f"[DONE] Saved: {out_path.resolve()}")


# ---------------------------------------------------------
# Webcam processing (✔ No freeze)
# ---------------------------------------------------------
def process_webcam(device=0):
    logger.info("Webcam mode started. Press Ctrl+C to stop.")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    out_path = VIDEO_OUT_DIR / f"webcam_{timestamp()}.mp4"
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

            detections = run_detection(frame)
            frame = draw_boxes(frame, detections)

            writer.write(frame)

    except KeyboardInterrupt:
        logger.info("Webcam stopped by user.")

    finally:
        cap.release()
        writer.release()
        logger.info(f"[DONE] Webcam video saved → {out_path.resolve()}")


# ---------------------------------------------------------
# Main router
# ---------------------------------------------------------
def detect_any(input_path: str):
    input_path = input_path.strip()

    if input_path.lower() in {"webcam", "cam"}:
        return process_webcam()

    path = Path(input_path)

    if path.is_file():
        if is_image(path):
            return process_image(path)
        elif is_video(path):
            return process_video(path)
        else:
            logger.error(f"Unsupported file type: {path.suffix}")
            return

    if path.is_dir():
        logger.info(f"Batch processing directory: {path}")
        for file in sorted(path.iterdir()):
            if is_image(file):
                process_image(file)
            elif is_video(file):
                process_video(file)
        return

    logger.error(f"Invalid path: {input_path}")


# if __name__ == "__main__":
#     detect_any("/Users/rolandoyax/Desktop/small.mp4")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        detect_any(sys.argv[1])
    else:
        print("Usage: python test_detect_any.py [image|video|directory|webcam]")
        print("Example:")
        print("   python test_detect_any.py webcam")
