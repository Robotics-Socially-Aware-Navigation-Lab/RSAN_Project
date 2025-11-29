"""
detect_any.py
-------------
Unified multimodal detection pipeline for RSAN.

Supports:
    - Image detection
    - Video detection
    - Webcam detection
    - Directory batch detection

Uses paths defined in:
    configs/project_paths.yaml

Outputs saved to:
    outputs/detections/images/
    outputs/detections/videos/
    outputs/detections/webcam/

Author: RSAN_Project_team
"""

import logging
import time
from pathlib import Path

import cv2

from src.perception.detect_utils import run_detection
from src.utils.file_utils import load_paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def is_image(file: Path) -> bool:
    return file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_video(file: Path) -> bool:
    return file.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Drawing Function
# ---------------------------------------------------------
def draw_detections(frame, detections):
    """Draw YOLO bounding boxes + class labels."""
    if detections is None:
        return frame

    for det in detections:
        try:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{det.class_name} {det.confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")

    return frame


# ---------------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------------
def process_image(path: Path):
    logger.info(f"Processing image: {path}")

    paths = load_paths()
    out_dir = paths["detections"] / "images"
    ensure_dir(out_dir)

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Unable to read image: {path}")
        return

    detections = run_detection(img)
    frame_with_boxes = draw_detections(img, detections)

    # Save output
    save_path = out_dir / f"{path.stem}_detected.jpg"
    cv2.imwrite(str(save_path), frame_with_boxes)
    logger.info(f"[SAVED] {save_path}")

    # Show result
    cv2.imshow("YOLO Image Detection", frame_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------
def process_video(path: Path):
    logger.info(f"Processing video: {path}")

    paths = load_paths()
    out_dir = paths["detections"] / "videos"
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = out_dir / f"{path.stem}_{timestamp}.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(3))
    h = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video.")
            break

        detections = run_detection(frame)
        frame = draw_detections(frame, detections)

        writer.write(frame)
        cv2.imshow("YOLO Video Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Video stopped by user.")
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"[SAVED] {save_path}")


# ---------------------------------------------------------
# WEBCAM PROCESSING
# ---------------------------------------------------------
def process_webcam(device=0):
    logger.info("Starting webcam (Press 'q' to exit)")

    paths = load_paths()
    out_dir = paths["detections"] / "webcam"
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        logger.error("Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Empty webcam frame.")
            continue

        detections = run_detection(frame)
        frame = draw_detections(frame, detections)

        cv2.imshow("YOLO Webcam Detection", frame)

        # Save one frame per second
        timestamp = int(time.time() * 1000)
        save_path = out_dir / f"webcam_{timestamp}.jpg"
        cv2.imwrite(str(save_path), frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Webcam stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------
def detect_any(input_path: str):
    input_path = input_path.strip()

    if input_path.lower() in {"webcam", "cam"}:
        return process_webcam()

    path = Path(input_path)

    if path.is_file():
        if is_image(path):
            return process_image(path)
        if is_video(path):
            return process_video(path)

        logger.error(f"Unsupported file type: {path.suffix}")
        return

    if path.is_dir():
        logger.info(f"Batch processing folder: {path}")
        for f in sorted(path.iterdir()):
            if is_image(f):
                process_image(f)
            elif is_video(f):
                process_video(f)
        return

    logger.error("Invalid input path. Must be a file, folder, or 'webcam'.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.perception.detect_any <image|video|directory|webcam>")
        sys.exit(1)

    detect_any(sys.argv[1])
