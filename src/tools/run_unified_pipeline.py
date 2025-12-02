"""
Unified RSAN Perception Pipeline (Rule-Based + Hybrid Fusion Version)
--------------------------------------------------------------------

Real-world robotics–grade behavior:
  • IMAGE mode  = one short rule-based summary sentence overlay
  • VIDEO mode  = one short rule-based summary sentence (clean HUD)
  • WEBCAM mode = one short rule-based summary sentence (real-time safe)
  • Full LLM reasoning always stored in outputs/reasoning_output.txt
  • Full scene + LLM context stored in outputs/full_pipeline/logs/results.json

Hybrid Fusion:
  • YOLO object detection refines indoor classification label
  • Indoor classifier predictions are corrected using detected objects

Author: students
Updated: students
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
from ultralytics import YOLO

from src.reasoning.indoor_classifier import IndoorClassifier
from src.reasoning.llm_reasoner import llm_reason_from_detections
from src.reasoning.scene_context import SceneContextResult, reason_about_scene
from src.utils.file_utils import load_paths

# ---------------------------------------------------------
# LOAD ROOM→OBJECT MAP
# ---------------------------------------------------------
ROOM_OBJECT_MAP = json.load(open("src/reasoning/room_object_map.json"))

# ---------------------------------------------------------
# PATHS
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
# UTIL
# ---------------------------------------------------------
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def is_video(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def append_json(entry: Dict[str, Any]) -> None:
    """Append a new entry to results.json safely."""
    if JSON_LOG.exists():
        with open(JSON_LOG, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(JSON_LOG, "w") as f:
        json.dump(data, f, indent=4)


def _extract_object_counts(det_result) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0

    if det_result is None or det_result.boxes is None:
        return counts, total

    names = det_result.names
    cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)

    for cid in cls_ids:
        label = names.get(int(cid), str(int(cid)))
        counts[label] += 1
        total += 1

    return counts, total


# ---------------------------------------------------------
# HYBRID FUSION LOGIC (Universal)
# ---------------------------------------------------------
def refine_scene_with_objects(scene_label: str, det_result) -> str:
    """
    Universal Hybrid Fusion:
    Uses object detection to refine indoor classification automatically.
    Works for unlimited scene classes.
    """
    if det_result is None or det_result.boxes is None:
        return scene_label

    names = det_result.names
    cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)
    detected = {names[int(cid)].lower() for cid in cls_ids}

    best_label = scene_label
    best_score = 0

    # Score based on matching objects
    for room, keywords in ROOM_OBJECT_MAP.items():
        score = len(set(keywords) & detected)
        if score > best_score:
            best_score = score
            best_label = room

    # Require at least 2 matching keywords
    if best_score >= 2:
        return best_label

    return scene_label


# ---------------------------------------------------------
# SUMMARY SENTENCE
# ---------------------------------------------------------
def _build_rule_based_summary(room_label: str, scene: SceneContextResult, det_result, llm_reasoning: str) -> str:

    room = room_label.lower()

    obj_counts, total_dets = _extract_object_counts(det_result)
    top_objs = [lbl for lbl, _ in obj_counts.most_common(2)]

    # Room phrase
    room_phrase = room if room else "scene"

    # Object phrase
    if top_objs and total_dets > 0:
        obj_fragment = f" with {', '.join(top_objs)}"
    elif total_dets > 0:
        obj_fragment = " with multiple detected objects"
    else:
        obj_fragment = ""

    # Crowd + risk
    crowd_fragment = f", crowd {scene.crowd_level}" if scene.crowd_level else ""
    risk_fragment = f", risk {scene.risk_score:.2f}"

    # Navigation hint
    hint_text = scene.navigation_hint.strip()
    first_sentence = hint_text.split(".", 1)[0].strip()
    hint_fragment = f", {first_sentence[0].lower() + first_sentence[1:]}" if first_sentence else ""

    summary = f"{room_phrase.capitalize()}{obj_fragment}{crowd_fragment}{risk_fragment}{hint_fragment}."

    if len(summary) > 160:
        summary = summary[:157].rstrip() + "..."

    return summary


# ---------------------------------------------------------
# DRAW OVERLAYS
# ---------------------------------------------------------
def draw_predictions(
    frame,
    det_result,
    scene_label: str,
    conf: float,
    scene: SceneContextResult,
    summary_sentence: Optional[str],
):
    img_h, img_w = frame.shape[:2]

    # YOLO bounding boxes
    if det_result is not None and det_result.boxes is not None:
        for box, cls_id, conf_box in zip(det_result.boxes.xyxy, det_result.boxes.cls, det_result.boxes.conf):

            x1, y1, x2, y2 = map(int, box.tolist())
            label = det_result.names[int(cls_id)]
            conf_val = float(conf_box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), max(2, img_w // 800))

            txt = f"{label} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame, txt, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Scene label
    cv2.putText(frame, f"{scene_label.upper()} ({conf:.2f})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

    # Crowd + risk
    cv2.putText(
        frame,
        f"Crowd: {scene.crowd_level} | Risk: {scene.risk_score:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 200),
        3,
    )

    # Navigation Hint
    cv2.putText(frame, f"Hint: {scene.navigation_hint}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 255), 3)

    # Summary sentence
    if summary_sentence:
        cv2.putText(frame, f"Summary: {summary_sentence}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (140, 0, 180), 3)

    return frame


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
def load_models():
    detector = YOLO(str(paths["models"] / "yolo_detector" / "best.pt"))
    classifier = IndoorClassifier(device="cpu")
    return detector, classifier


# ---------------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------------
def process_image(path: Path, detector, classifier):
    logger.info(f"[IMAGE] {path}")

    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Could not read image: {path}")
        return

    det_result = detector(img)[0]

    cls_raw = classifier.predict(img)
    raw_scene_label = cls_raw.label

    fused_scene_label = refine_scene_with_objects(raw_scene_label, det_result)

    scene = reason_about_scene(fused_scene_label, det_result)

    full_llm = llm_reason_from_detections(
        room_label=fused_scene_label,
        detections=det_result,
        source=str(path),
        save=True,
        print_to_console=True,
    )

    summary_sentence = _build_rule_based_summary(fused_scene_label, scene, det_result, full_llm)

    frame = draw_predictions(
        img,
        det_result,
        fused_scene_label,
        cls_raw.confidence,
        scene,
        summary_sentence,
    )

    out_path = IMG_OUT / f"{path.stem}_full_{timestamp()}.jpg"
    cv2.imwrite(str(out_path), frame)

    obj_counts, total_dets = _extract_object_counts(det_result)

    append_json(
        {
            "mode": "image",
            "input": str(path.resolve()),
            "output": str(out_path.resolve()),
            "timestamp": timestamp(),
            "room_label_raw": raw_scene_label,
            "room_label_fused": fused_scene_label,
            "room_confidence": cls_raw.confidence,
            "crowd_level": scene.crowd_level,
            "risk_score": scene.risk_score,
            "navigation_hint": scene.navigation_hint,
            "total_detections": total_dets,
            "object_counts": dict(obj_counts),
            "llm_full_reasoning": full_llm,
            "summary_sentence": summary_sentence,
        }
    )

    logger.info(f"Saved → {out_path}")


# ---------------------------------------------------------
# VIDEO PROCESSING
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

    ret, first_frame = cap.read()
    if not ret:
        logger.error("Could not read initial frame.")
        return

    det_first = detector(first_frame)[0]

    cls_raw = classifier.predict(first_frame)
    raw_scene_label = cls_raw.label

    fused_label = refine_scene_with_objects(raw_scene_label, det_first)
    scene_first = reason_about_scene(fused_label, det_first)

    full_llm = llm_reason_from_detections(
        room_label=fused_label,
        detections=det_first,
        source=str(path),
        save=True,
        print_to_console=True,
    )

    summary_sentence = _build_rule_based_summary(fused_label, scene_first, det_first, full_llm)

    frame = draw_predictions(first_frame, det_first, fused_label, cls_raw.confidence, scene_first, summary_sentence)
    writer.write(frame)

    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_result = detector(frame)[0]
        cls_raw_f = classifier.predict(frame)

        fused_scene_label_f = refine_scene_with_objects(cls_raw_f.label, det_result)
        scene_f = reason_about_scene(fused_scene_label_f, det_result)

        frame = draw_predictions(
            frame, det_result, fused_scene_label_f, cls_raw_f.confidence, scene_f, summary_sentence
        )

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    obj_counts, total_dets = _extract_object_counts(det_first)

    append_json(
        {
            "mode": "video",
            "input": str(path.resolve()),
            "output": str(out_path.resolve()),
            "timestamp": timestamp(),
            "frames": frame_count,
            "room_label_raw_first_frame": raw_scene_label,
            "room_label_fused_first_frame": fused_label,
            "room_confidence_first_frame": cls_raw.confidence,
            "crowd_level_first_frame": scene_first.crowd_level,
            "risk_score_first_frame": scene_first.risk_score,
            "navigation_hint_first_frame": scene_first.navigation_hint,
            "total_detections_first_frame": total_dets,
            "object_counts_first_frame": dict(obj_counts),
            "llm_full_reasoning": full_llm,
            "summary_sentence": summary_sentence,
        }
    )

    logger.info(f"Saved video → {out_path}")


# ---------------------------------------------------------
# WEBCAM PROCESSING
# ---------------------------------------------------------
def process_webcam(detector, classifier, index: int = 0):
    logger.info("[WEBCAM] Starting webcam (Ctrl+C to stop)")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Webcam cannot be accessed.")
        return

    out_path = VID_OUT / f"webcam_full_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    ret, first_frame = cap.read()
    if not ret:
        logger.error("Could not read first webcam frame.")
        return

    det_first = detector(first_frame)[0]
    cls_raw = classifier.predict(first_frame)
    raw_scene_label = cls_raw.label

    fused_label = refine_scene_with_objects(raw_scene_label, det_first)
    scene_first = reason_about_scene(fused_label, det_first)

    full_llm = llm_reason_from_detections(
        room_label=fused_label,
        detections=det_first,
        source="webcam",
        save=True,
        print_to_console=False,
    )

    summary_sentence = _build_rule_based_summary(fused_label, scene_first, det_first, full_llm)

    frame = draw_predictions(first_frame, det_first, fused_label, cls_raw.confidence, scene_first, summary_sentence)
    writer.write(frame)
    cv2.imshow("Unified Pipeline Webcam", frame)

    frame_count = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            det_result = detector(frame)[0]
            cls_raw_f = classifier.predict(frame)

            fused_scene_label = refine_scene_with_objects(cls_raw_f.label, det_result)
            scene_f = reason_about_scene(fused_scene_label, det_result)

            frame = draw_predictions(
                frame, det_result, fused_scene_label, cls_raw_f.confidence, scene_f, summary_sentence
            )

            writer.write(frame)
            cv2.imshow("Unified Pipeline Webcam", frame)
            frame_count += 1

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Webcam stopped by user.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    obj_counts, total_dets = _extract_object_counts(det_first)

    append_json(
        {
            "mode": "webcam",
            "input": f"webcam_index_{index}",
            "output": str(out_path.resolve()),
            "timestamp": timestamp(),
            "frames": frame_count,
            "room_label_raw_first_frame": raw_scene_label,
            "room_label_fused_first_frame": fused_label,
            "room_confidence_first_frame": cls_raw.confidence,
            "crowd_level_first_frame": scene_first.crowd_level,
            "risk_score_first_frame": scene_first.risk_score,
            "navigation_hint_first_frame": scene_first.navigation_hint,
            "total_detections_first_frame": total_dets,
            "object_counts_first_frame": dict(obj_counts),
            "llm_full_reasoning": full_llm,
            "summary_sentence": summary_sentence,
        }
    )

    logger.info(f"Saved webcam → {out_path}")


# ---------------------------------------------------------
# MAIN ROUTER
# ---------------------------------------------------------
def run_unified(input_path: str):
    detector, classifier = load_models()
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


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_unified(sys.argv[1])
    else:
        print("Usage: python -m src.tools.run_unified_pipeline <image|video|folder|webcam>")
