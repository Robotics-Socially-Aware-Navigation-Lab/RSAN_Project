"""
src/perception/detect_objects.py

High-level YOLOv8 object detection interface for the RSAN project.

Responsibilities:
- Load the YOLO detection model from models/yolo_detector/best.pt
- Provide a clean, typed API to run detection on images (np.ndarray)
- Optional visualization utilities and simple CLI for debugging

This module is designed to be imported by:
- src/reasoning/scene_context.py
- ROS2 bridge code (ros2_interface / san_node)
- Unit tests in src/tests/test_yolo_pipeline.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic fallback config in case project logger isn't configured yet
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------

# Project root: .../RSAN_Project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default detector model path
DEFAULT_DETECTOR_PATH = PROJECT_ROOT / "models" / "yolo_detector" / "best.pt"


@dataclass
class Detection:
    """
    A single detection result from YOLOv8.

    Attributes:
        class_id: Integer class index.
        class_name: Human-readable label (from YOLO model names mapping).
        confidence: Confidence score in [0, 1].
        bbox_xyxy: Bounding box in (x1, y1, x2, y2) pixel coords.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


# ------------------------------------------------------------
# Detector class
# ------------------------------------------------------------


class ObjectDetector:
    """
    Wrapper around a YOLOv8 detection model.

    Usage:
        detector = ObjectDetector()  # uses default model path
        detections = detector.detect(frame)

    Designed to be:
    - Imported once and reused (model kept in memory)
    - Easy to swap model path or thresholds (configurable)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: Path to YOLOv8 .pt weights file. If None, uses DEFAULT_DETECTOR_PATH.
            conf_threshold: Minimum confidence score to keep a detection.
            iou_threshold: IOU threshold used internally by YOLO for NMS.
            device: Optional device string for YOLO (e.g., 'cpu', 'cuda', 'cuda:0').
                    If None, YOLO auto-selects.
        """
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_DETECTOR_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if not self.model_path.is_file():
            msg = f"Detector model file not found at: {self.model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Loading YOLO detector model from %s", self.model_path)
        # NOTE: YOLO will auto-detect device if device is None
        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run object detection on a single BGR image (OpenCV format).

        Args:
            image_bgr: HxWx3 uint8 image in BGR color space.
            conf_threshold: Optional override of the default confidence threshold.

        Returns:
            A list of Detection objects.
        """
        if image_bgr is None or image_bgr.size == 0:
            logger.warning("Received empty image for detection; returning no detections.")
            return []

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Run YOLO inference. This returns a list-like of Results.
        results = self.model(
            image_bgr,
            conf=threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                class_name = self.model.names.get(class_id, str(class_id))

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox_xyxy=bbox,
                    )
                )

        return detections

    def draw_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
        show_label: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the image.

        Args:
            image_bgr: Original BGR image.
            detections: List of Detection objects.
            show_label: Whether to overlay class name and confidence.

        Returns:
            Annotated image (new np.ndarray).
        """
        output = image_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if show_label:
                label = f"{det.class_name} {det.confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    output,
                    (x1, y1 - th - baseline),
                    (x1 + tw, y1),
                    (0, 255, 0),
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    output,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return output


# ------------------------------------------------------------
# Convenience functions for other modules
# ------------------------------------------------------------

# Optional global singleton detector so other modules can just call `get_default_detector()`
_default_detector: Optional[ObjectDetector] = None


def get_default_detector() -> ObjectDetector:
    """
    Lazily initialize and return a global ObjectDetector instance.

    This avoids reloading the YOLO model multiple times across the project.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = ObjectDetector()
    return _default_detector


def detect_objects(
    image_bgr: np.ndarray,
    conf_threshold: Optional[float] = None,
) -> List[Detection]:
    """
    Functional-style wrapper around the default detector.

    Args:
        image_bgr: HxWx3 BGR image.
        conf_threshold: Optional override for confidence threshold.

    Returns:
        List of Detection objects.
    """
    detector = get_default_detector()
    return detector.detect(image_bgr, conf_threshold=conf_threshold)


# ------------------------------------------------------------
# CLI / manual testing
# ------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on an image or webcam.")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to image/video file or webcam index (default: '0' for webcam).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display window (useful on headless systems).",
    )
    return parser.parse_args()


def _run_cli() -> None:
    args = _parse_args()

    # Initialize detector
    detector = ObjectDetector(conf_threshold=args.conf)

    # Webcam vs file
    if args.source.isdigit():
        # Webcam mode
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            logger.error("Could not open webcam index %s", args.source)
            return

        logger.info("Starting webcam detection (press 'q' to quit)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam.")
                break

            dets = detector.detect(frame)
            vis = detector.draw_detections(frame, dets)

            if not args.no_show:
                cv2.imshow("YOLOv8 Detector", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
    else:
        # Image or video file
        src_path = Path(args.source)
        if not src_path.is_file():
            logger.error("Source file not found: %s", src_path)
            return

        # Try to read as image first
        img = cv2.imread(str(src_path))
        if img is not None:
            logger.info("Running detection on image: %s", src_path)
            dets = detector.detect(img)
            for d in dets:
                logger.info("Detected %s (%.2f) at %s", d.class_name, d.confidence, d.bbox_xyxy)
            vis = detector.draw_detections(img, dets)
            if not args.no_show:
                cv2.imshow("YOLOv8 Detector", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            # Fallback: treat as video
            cap = cv2.VideoCapture(str(src_path))
            if not cap.isOpened():
                logger.error("Could not open video: %s", src_path)
                return
            logger.info("Running detection on video: %s (press 'q' to quit)", src_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                dets = detector.detect(frame)
                vis = detector.draw_detections(frame, dets)
                if not args.no_show:
                    cv2.imshow("YOLOv8 Detector", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_cli()
