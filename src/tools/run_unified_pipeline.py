"""
Unified RSAN Perception Pipeline (Senior Engineer Version)
----------------------------------------------------------

This refactored pipeline integrates:

    ✔ Clean ObjectDetector API (ALL detections flow through one source)
    ✔ IndoorClassifier (ResNet / Places365 / MIT head)
    ✔ Hybrid scene fusion (object evidence + classifier confidence)
    ✔ Symbolic + LLM reasoning
    ✔ Robust annotation for images, videos, webcam
    ✔ ROS2-friendly modular structure

Objectives achieved:
    - No raw YOLO calls
    - ALLOWED_NAMES and KEEP_IDS filtering IS applied
    - Consistent Detection dataclass across whole system
    - Team-safe (no API key → fallback reasoning)
    - Scalable, maintainable, industry-standard architecture
"""

from __future__ import annotations

# ================================================================
# ALL IMPORTS MUST BE HERE
# ================================================================

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import cv2

# --- RSAN imports ---
from src.perception.detect_objects import get_default_detector, Detection
from src.reasoning.indoor_classifier import IndoorClassifier
from src.reasoning.scene_context import reason_about_scene
from src.reasoning.llm_reasoner import llm_reason_from_detections
from src.utils.file_utils import load_paths

# ----------------------------------------------------------------------
# Load object-room map (Hybrid Fusion)
# ----------------------------------------------------------------------
ROOM_OBJECT_MAP = json.load(open("src/reasoning/room_object_map.json"))

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


# =============================================================
# Utility functions
# =============================================================
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_video(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def append_json(entry: Dict[str, Any]):
    if JSON_LOG.exists():
        data = json.load(open(JSON_LOG))
    else:
        data = []
    data.append(entry)
    json.dump(data, open(JSON_LOG, "w"), indent=4)


# =============================================================
# Hybrid Fusion (refined scene reasoning)
# =============================================================
def refine_scene_with_objects(
    raw_label: str,
    detections: List[Detection],
    cls_confidence: float,
) -> str:
    """Use object evidence to refine scene label."""
    if not detections:
        return raw_label

    detected = {d.class_name.lower() for d in detections}

    best_label = raw_label
    best_score = cls_confidence

    for room, keywords in ROOM_OBJECT_MAP.items():
        match_count = len(detected & set(k.lower() for k in keywords))
        object_score = match_count / max(1, len(keywords))

        scene_score = 0.6 * cls_confidence + 0.4 * object_score

        if scene_score > best_score:
            best_label = room
            best_score = scene_score

    return best_label


# =============================================================
# Draw detections and predictions
# =============================================================
def draw_detections(frame, detections: List[Detection]):
    """Draw bounding boxes for Detection dataclass."""

    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        label_text = f"{d.class_name} {d.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        (tw, th), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            2,
        )

        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 10, y1),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

    return frame


def draw_scene_annotations(frame, scene_label, cls_conf, scene_info, summary_sentence):
    cv2.putText(
        frame,
        f"{scene_label.upper()} ({cls_conf:.2f})",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 255),
        4,
    )

    cv2.putText(
        frame,
        f"Crowd: {scene_info.crowd_level} | Risk: {scene_info.risk_score:.2f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 50, 255),
        3,
    )

    cv2.putText(
        frame,
        f"Hint: {scene_info.navigation_hint}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (100, 0, 255),
        3,
    )

    if summary_sentence:
        cv2.putText(
            frame,
            f"{summary_sentence[:120]}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (150, 0, 150),
            2,
        )

    return frame


# =============================================================
# Main processing (image)
# =============================================================
def process_image(path: Path, detector, classifier):
    logger.info(f"[IMAGE] {path}")
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Image read failed.")
        return

    detections = detector.detect(img)
    cls = classifier.predict(img)

    fused_label = refine_scene_with_objects(cls.label, detections, cls.confidence)
    scene = reason_about_scene(fused_label, detections)

    summary = llm_reason_from_detections(fused_label, detections, source=str(path))

    annotated = img.copy()
    annotated = draw_detections(annotated, detections)
    annotated = draw_scene_annotations(annotated, fused_label, cls.confidence, scene, summary)

    out = IMG_OUT / f"{path.stem}_full_{timestamp()}.jpg"
    cv2.imwrite(str(out), annotated)
    logger.info(f"Saved → {out}")


# =============================================================
# Main processing (video)
# =============================================================
def process_video(path: Path, detector, classifier):
    logger.info(f"[VIDEO] {path}")
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        logger.error("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(3)), int(cap.get(4))
    out_path = VID_OUT / f"{path.stem}_full_{timestamp()}.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    ret, first = cap.read()
    if not ret:
        return

    det_first = detector.detect(first)
    cls_first = classifier.predict(first)

    fused_first = refine_scene_with_objects(cls_first.label, det_first, cls_first.confidence)
    scene_first = reason_about_scene(fused_first, det_first)

    summary = llm_reason_from_detections(fused_first, det_first, source=str(path))

    frame = first.copy()
    frame = draw_detections(frame, det_first)
    frame = draw_scene_annotations(frame, fused_first, cls_first.confidence, scene_first, summary)
    writer.write(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detector.detect(frame)
        cls = classifier.predict(frame)
        fused = refine_scene_with_objects(cls.label, det, cls.confidence)
        scene = reason_about_scene(fused, det)

        frame = draw_detections(frame, det)
        frame = draw_scene_annotations(frame, fused, cls.confidence, scene, summary)
        writer.write(frame)

    cap.release()
    writer.release()
    logger.info(f"Saved video → {out_path}")


# =============================================================
# Webcam processing
# =============================================================
def process_webcam(detector, classifier, index=0):
    logger.info("[WEBCAM] Starting webcam...")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Webcam unavailable.")
        return

    ret, first = cap.read()
    if not ret:
        logger.error("No frames received from webcam.")
        return

    det_first = detector.detect(first)
    cls_first = classifier.predict(first)

    fused_first = refine_scene_with_objects(cls_first.label, det_first, cls_first.confidence)
    # scene_first = reason_about_scene(fused_first, det_first)

    summary = llm_reason_from_detections(
        fused_first,
        det_first,
        source="webcam",
        print_to_console=False,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        det = detector.detect(frame)
        cls = classifier.predict(frame)

        fused = refine_scene_with_objects(cls.label, det, cls.confidence)
        scene = reason_about_scene(fused, det)

        frame = draw_detections(frame, det)
        frame = draw_scene_annotations(frame, fused, cls.confidence, scene, summary)

        cv2.imshow("RSAN Unified Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================
# Router
# =============================================================
def run_unified(input_path: str):
    detector = get_default_detector()
    classifier = IndoorClassifier()

    p = Path(input_path)

    if input_path.lower() in {"webcam", "cam"}:
        return process_webcam(detector, classifier)

    if p.is_file() and is_image(p):
        return process_image(p, detector, classifier)

    if p.is_file() and is_video(p):
        return process_video(p, detector, classifier)

    if p.is_dir():
        for f in sorted(p.iterdir()):
            if is_image(f):
                process_image(f, detector, classifier)
            elif is_video(f):
                process_video(f, detector, classifier)
        return

    logger.error(f"Invalid input: {input_path}")


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_unified(sys.argv[1])
    else:
        print("Usage: python -m src.tools.run_unified_pipeline <image|video|folder|webcam>")
