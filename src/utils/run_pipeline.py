"""
End-to-end RSAN perception pipeline (standalone).

Webcam → YOLOv8 object detector → Indoor scene classifier → Scene reasoning
→ Annotated OpenCV window.

Run from project root with:

    conda activate rsan_env
    python -m src.utils.run_pipeline --camera 0

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO

from src.reasoning.indoor_classifier import IndoorClassifier
from src.reasoning.scene_context import reason_about_scene
from src.utils.logger import get_logger
from src.perception.detector_config import KEEP_IDS

log = get_logger(__name__)


# ---------------------------------------------------------
# Path Resolution — **THIS FIXES YOUR MODEL PATH BUG**
# ---------------------------------------------------------
def _get_project_root() -> Path:
    """
    Correct project root resolver.

    run_pipeline.py lives in:
        RSAN_Project/src/utils/run_pipeline.py

    parents[0] = utils/
    parents[1] = src/
    parents[2] = RSAN_Project/   ← correct root
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Detector Loader
# ---------------------------------------------------------
def _load_detector(model_path: Path | None = None) -> YOLO:
    root = _get_project_root()

    # If user didn't specify model_path, load from RSAN models directory
    if model_path is None:
        model_path = root / "models" / "yolo_detector" / "best.pt"
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO detector not found at:\n  {model_path}\n"
            "Make sure the file exists inside: models/yolo_detector/best.pt"
        )

    log.info("Loading YOLOv8 detector from %s", model_path)
    return YOLO(str(model_path))


# ---------------------------------------------------------
# Utility: Extract object names
# ---------------------------------------------------------
def _extract_object_names_from_results(result) -> List[str]:
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names
    return [names[int(i)] for i in cls_ids]


# ---------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------
def run_pipeline(
    camera_index: int = 0,
    detector_path: Path | None = None,
    display: bool = True,
) -> None:

    root = _get_project_root()
    log.info("RSAN pipeline starting. project_root = %s", root)

    # Load YOLOv8 detector
    detector = _load_detector(detector_path)

    # Load indoor classifier
    classifier = IndoorClassifier()

    # Start camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    log.info("Camera %d opened successfully.", camera_index)
    log.info("Press 'q' to exit.")

    prev_time = time.time()
    fps_smooth = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame from camera. Exiting.")
                break

            # Safety check
            if frame is None or frame.size == 0:
                log.warning("Empty frame received; skipping.")
                continue

            # 1) Object detection
            try:
                det_results = detector(frame, classes=KEEP_IDS)[0]
            except Exception as exc:
                log.error("Detector inference failed: %s", exc, exc_info=True)
                continue

            # 2) Indoor classification
            try:
                cls_result = classifier.predict(frame)
            except Exception as exc:
                log.error("Classifier inference failed: %s", exc, exc_info=True)
                continue

            # 3) Scene reasoning
            scene = reason_about_scene(cls_result.label, det_results)

            # 4) FPS calculation
            now = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if fps_smooth is None:
                fps_smooth = inst_fps
            else:
                # smooth FPS value for stable reading
                fps_smooth = (0.9 * fps_smooth) + (0.1 * inst_fps)

            # 5) Render annotated frame
            if display:
                # Draw detection bounding boxes
                for box, cls_id in zip(det_results.boxes.xyxy, det_results.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label = det_results.names[int(cls_id)]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Overlay pipeline info on screen
                info = [
                    f"Room: {cls_result.label} ({cls_result.confidence:.2f})",
                    f"Crowd: {scene.crowd_level} | Risk: {scene.risk_score:.2f}",
                    f"Hint: {scene.navigation_hint}",
                    f"FPS: {fps_smooth:.1f}",
                ]

                y = 25
                for line in info:
                    cv2.putText(
                        frame,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 25

                cv2.imshow("RSAN Pipeline", frame)

                # Exit key
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    log.info("User pressed 'q'. Exiting.")
                    break

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, shutting down...")

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        log.info("RSAN pipeline stopped.")


# ---------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSAN perception pipeline")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default = 0)",
    )
    parser.add_argument(
        "--detector-path",
        type=str,
        default=None,
        help="Custom YOLO detector path (optional)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without OpenCV display window",
    )
    return parser.parse_args()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    detector_path = Path(args.detector_path) if args.detector_path else None

    run_pipeline(
        camera_index=args.camera,
        detector_path=detector_path,
        display=not args.no_display,
    )
