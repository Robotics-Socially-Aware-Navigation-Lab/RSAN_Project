"""
detect_any.py
-------------
Unified multimodal detection pipeline for RSAN.

Supports:
    - Single image
    - Multiple images
    - Single video
    - Multiple videos
    - Webcam streaming

Fully robust to:
    - Missing files
    - Corrupt images
    - Unsupported formats
    - Empty frames
    - Broken video streams
    - Keyboard interrupts

Author: Senior ML Engineer / Robotics Navigation Specialist
"""

import os
import cv2
import logging
from pathlib import Path
from typing import List
from src.perception.detect_utils import run_detection

# ---------------------------------------------------------
# Logging Configuration (Production Standard)
# ---------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_image(file: Path) -> bool:
    return file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_video(file: Path) -> bool:
    return file.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


# ---------------------------------------------------------
# DRAWING UTILITIES (Scalable + Modular)
# ---------------------------------------------------------
def draw_detections(frame, detections):
    """Draw bounding boxes + labels safely."""
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

    if not path.exists():
        logger.error(f"Image does not exist: {path}")
        return

    img = cv2.imread(str(path))

    if img is None:
        logger.error(f"Unable to read image (corrupted?): {path}")
        return

    detections = run_detection(img)

    frame_with_boxes = draw_detections(img, detections)

    cv2.imshow("YOLO Image Detection", frame_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------
def process_video(path: Path):
    logger.info(f"Processing video: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Reached end of video.")
                break

            detections = run_detection(frame)
            frame = draw_detections(frame, detections)

            cv2.imshow("YOLO Video Detection", frame)

            # Quit video with Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User exited video playback.")
                break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------
# LIVE WEBCAM STREAM
# ---------------------------------------------------------
def process_webcam(device=0):
    logger.info("Starting webcam (Press 'q' to exit).")

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        logger.error("Could not access webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Empty webcam frame — continuing.")
                continue

            detections = run_detection(frame)
            frame = draw_detections(frame, detections)

            cv2.imshow("YOLO Webcam Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Webcam stream terminated by user.")
                break

    except KeyboardInterrupt:
        logger.warning("Webcam interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------
# UNIFIED PIPELINE
# ---------------------------------------------------------
def detect_any(input_path: str):
    """Smart router that selects appropriate detection pipeline."""

    input_path = input_path.strip()

    # Webcam mode
    if input_path.lower() in {"webcam", "cam"}:
        return process_webcam()

    path = Path(input_path)

    # Single file
    if path.is_file():
        if is_image(path):
            return process_image(path)
        elif is_video(path):
            return process_video(path)
        else:
            logger.error(f"Unsupported file type: {path.suffix}")
            return

    # Directory — batch processing
    if path.is_dir():
        logger.info(f"Batch processing directory: {path}")

        files = sorted(path.iterdir())
        if not files:
            logger.error("Directory is empty — nothing to process.")
            return

        for file in files:
            if is_image(file):
                process_image(file)
            elif is_video(file):
                process_video(file)

        return

    logger.error("Invalid input path. Must be file, directory, or 'webcam'.")
    return


# ---------------------------------------------------------
# MAIN (for standalone debugging)
# ---------------------------------------------------------
if __name__ == "__main__":
    detect_any("/Users/rolandoyax/Desktop/IMG_0025.jpg")
