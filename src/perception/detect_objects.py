"""
src/perception/detect_objects.py

High-level YOLOv8 object detection interface for the RSAN project.
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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

# ------------------------------------------------------------
# CLASS FILTERS
# ------------------------------------------------------------

# (Optional) keep only these IDs at the YOLO level
KEEP_IDS = [
    0,
    15,
    16,
    24,
    25,
    26,
    28,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    82,
    83,
    85,
    86,
    88,
    89,
    90,
    91,
]

# ✅ This is the *hard* gate: ONLY these class names are allowed to appear.
# Remove or add as needed for RSAN.



# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR_PATH = PROJECT_ROOT / "models" / "yolo_detector" / "best.pt"


@dataclass
class Detection:
    """
    A single detection result from YOLOv8.
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
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:

        self.model_path = Path(model_path) if model_path is not None else DEFAULT_DETECTOR_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if not self.model_path.is_file():
            msg = f"Detector model file not found at: {self.model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Loading YOLO detector model from %s", self.model_path)
        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:

        if image_bgr is None or image_bgr.size == 0:
            logger.warning("Received empty image for detection; returning no detections.")
            return []

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # ------------------------------------------------------------
        # YOLO INFERENCE (with optional ID filter)
        # ------------------------------------------------------------
        results = self.model(
            image_bgr,
            conf=threshold,
            iou=self.iou_threshold,
            classes=KEEP_IDS if KEEP_IDS else None,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                class_name = self.model.names.get(class_id, str(class_id))

                # 🔍 Debug (optional): see what IDs/names are coming out
                # logger.info("YOLO raw: id=%s name=%s conf=%.2f", class_id, class_name, conf)



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

        output = image_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
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
# Singleton for shared detector instance
# ------------------------------------------------------------

_default_detector: Optional[ObjectDetector] = None


def get_default_detector() -> ObjectDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = ObjectDetector()
    return _default_detector


def detect_objects(
    image_bgr: np.ndarray,
    conf_threshold: Optional[float] = None,
) -> List[Detection]:
    detector = get_default_detector()
    return detector.detect(image_bgr, conf_threshold=conf_threshold)


# ------------------------------------------------------------
# CLI for manual testing
# ------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on an image or webcam.")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _run_cli() -> None:
    args = _parse_args()
    detector = ObjectDetector(conf_threshold=args.conf)

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            logger.error("Could not open webcam index %s", args.source)
            return

        logger.info("Starting webcam detection (press 'q' to quit)...")
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

    else:
        src_path = Path(args.source)
        img = cv2.imread(str(src_path))
        if img is not None:
            dets = detector.detect(img)
            for d in dets:
                logger.info("Detected %s (%.2f) at %s", d.class_name, d.confidence, d.bbox_xyxy)
            vis = detector.draw_detections(img, dets)
            if not args.no_show:
                cv2.imshow("YOLOv8 Detector", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_cli()
