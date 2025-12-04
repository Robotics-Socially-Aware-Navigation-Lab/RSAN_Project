# # [CHANGED] Updated description to reflect ResNet Places365 indoor classifier (not YOLOv8-CLS).
# """
# Unified RSAN Perception Pipeline (Hybrid + Safe LLM Version)
# --------------------------------------------------------------------

# Capabilities:
#   • YOLOv8 object detection
#   • Indoor scene classification (ResNet Places365 / MIT Indoor head)
#   • Hybrid Fusion (YOLO objects refine indoor classification)
#   • Scene-level reasoning (risk, crowd, hints)
#   • Optional LLM reasoning (safe: disabled if no API key)
#   • Clean 1-sentence summary overlay
#   • Image, video, and webcam support

# Team-Safe Behavior:
#   • If OPENAI_API_KEY is missing → pipeline runs normally
#   • LLM reasoning becomes fallback text
#   • Red overlay: “LLM DISABLED — No API Key”
# """

# from __future__ import annotations

# import json
# import logging
# from collections import Counter
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, Optional, Tuple

# import cv2
# from ultralytics import YOLO

# from src.reasoning.indoor_classifier import IndoorClassifier
# from src.reasoning.llm_reasoner import llm_reason_from_detections
# from src.reasoning.scene_context import SceneContextResult, reason_about_scene
# from src.utils.file_utils import load_paths


# # ---------------------------------------------------------
# # LOAD ROOM→OBJECT MAP
# # ---------------------------------------------------------
# ROOM_OBJECT_MAP = json.load(open("src/reasoning/room_object_map.json"))


# # ---------------------------------------------------------
# # PIPELINE PATHS
# # ---------------------------------------------------------
# paths = load_paths()

# ROOT = paths["full_pipeline"]
# IMG_OUT = ROOT / "images"
# VID_OUT = ROOT / "videos"
# LOG_DIR = ROOT / "logs"

# IMG_OUT.mkdir(parents=True, exist_ok=True)
# VID_OUT.mkdir(parents=True, exist_ok=True)
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# JSON_LOG = LOG_DIR / "results.json"

# logger = logging.getLogger("unified_pipeline")
# logger.setLevel(logging.INFO)


# # ---------------------------------------------------------
# # SIMPLE UTILITIES
# # ---------------------------------------------------------
# def timestamp() -> str:
#     return datetime.now().strftime("%Y%m%d_%H%M%S")


# def is_image(p: Path) -> bool:
#     return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


# def is_video(p: Path) -> bool:
#     return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


# def append_json(entry: Dict[str, Any]) -> None:
#     """Append a new entry to results.json safely."""
#     if JSON_LOG.exists():
#         data = json.load(open(JSON_LOG))
#     else:
#         data = []
#     data.append(entry)
#     json.dump(data, open(JSON_LOG, "w"), indent=4)


# def _extract_object_counts(det_result) -> Tuple[Counter, int]:
#     """Convert YOLO detection results → a count dict."""
#     counts = Counter()
#     total = 0

#     if det_result is None or det_result.boxes is None:
#         return counts, 0

#     names = det_result.names
#     cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)

#     for cid in cls_ids:
#         label = names.get(int(cid), f"id_{cid}")
#         counts[label] += 1
#         total += 1

#     return counts, total


# # ---------------------------------------------------------
# # HYBRID FUSION (YOLO + Indoor CLS)
# # ---------------------------------------------------------
# def refine_scene_with_objects(raw_label: str, det_result, cls_confidence: float) -> str:
#     """
#     HYBRID FUSION (UPGRADED VERSION):
#     Combines object evidence + classifier confidence using:

#         scene_score = classifier_confidence * 0.6 + object_match_score * 0.4

#     This gives smooth, stable, probabilistic refinement instead of the old
#     "2-object rule" which was too rigid.
#     """

#     # ----------------------------------------------------------------------
#     # ORIGINAL VERSION (kept for reference but now DISABLED)
#     # ----------------------------------------------------------------------
#     # if det_result is None or det_result.boxes is None:
#     #     return raw_label
#     #
#     # names = det_result.names
#     # cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)
#     # detected = {names[int(cid)].lower() for cid in cls_ids}
#     #
#     # best_label = raw_label
#     # best_score = 0
#     #
#     # for room, keywords in ROOM_OBJECT_MAP.items():
#     #     score = len(detected & set(keywords))
#     #     if score > best_score:
#     #         best_score = score
#     #         best_label = room
#     #
#     # return best_label if best_score >= 2 else raw_label
#     #
#     # WHY THIS WAS REPLACED:
#     # - Hard rule: required 2 matching objects.
#     # - No use of classifier confidence.
#     # - Could fail on scenes with few objects (hallways, bathrooms, etc.).
#     # ----------------------------------------------------------------------

#     # ----------------------------------------------------------------------
#     # NEW VERSION — Probabilistic Hybrid Fusion
#     # ----------------------------------------------------------------------

#     # If no YOLO detections, return classifier prediction unchanged.
#     if det_result is None or det_result.boxes is None:
#         return raw_label

#     # Extract detected YOLO object names
#     names = det_result.names
#     cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)
#     detected = {names[int(cid)].lower() for cid in cls_ids}

#     best_label = raw_label
#     best_scene_score = cls_confidence  # baseline score

#     for room, keywords in ROOM_OBJECT_MAP.items():

#         # Count matching objects
#         matches = len(detected & set(keywords))

#         # Normalize the object score to 0–1
#         object_match_score = matches / max(1, len(keywords))

#         # Weighted probabilistic fusion
#         scene_score = cls_confidence * 0.6 + object_match_score * 0.4

#         # Keep the best room found
#         if scene_score > best_scene_score:
#             best_scene_score = scene_score
#             best_label = room

#     return best_label

# # def refine_scene_with_objects(raw_label: str, det_result) -> str:
# #     """
# #     HYBRID FUSION: Improve indoor scene classification using YOLO detections.

# #     WHY THIS EXISTS:
# #         YOLO-based scene classifiers sometimes mislabel scenes because they only
# #         see the entire image as a single classification problem. But YOLO object
# #         detection provides fine-grained clues about what objects are actually present.

# #         Example:
# #             Indoor classifier predicts: "hallway" (weak confidence)
# #             YOLO detects: chair, desk, whiteboard → This looks like a CLASSROOM.

# #         This function uses a JSON mapping (ROOM_OBJECT_MAP) to match detected
# #         objects to typical room types and correct the classifier output.

# #     HOW IT WORKS:
# #         1. Extract all detected object names from YOLO.
# #         2. For each room type (office, classroom, hallway, etc.):
# #                Count how many of its expected objects appear in the frame.
# #         3. Pick the room with the highest match score.
# #         4. Only override the classifier label if at least TWO objects match.
# #            (This avoids random single-object misfires.)

# #     PARAMETERS:
# #         raw_label  → The original room prediction from the indoor classifier.
# #         det_result → YOLO detection result, containing bounding boxes & classes.

# #     RETURNS:
# #         A refined room label (string). May be the same as raw_label if
# #         object evidence is insufficient.
# #     """

# #     # Safety check: if YOLO returned nothing, keep the original label.
# #     if det_result is None or det_result.boxes is None:
# #         return raw_label

# #     # Extract YOLO class names (e.g., "chair", "tv") for each detected object.
# #     names = det_result.names
# #     cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)

# #     # Convert detected object class IDs → lowercase name set for fast matching.
# #     detected = {names[int(cid)].lower() for cid in cls_ids}

# #     # Initialize fusion variables: best guess begins as the raw classifier label.
# #     best_label = raw_label
# #     best_score = 0

# #     # ROOM_OBJECT_MAP: dict where each room has a list of typical objects.
# #     # Example:
# #     #   "office": ["desk", "chair", "monitor"]
# #     #   "kitchen": ["oven", "microwave", "sink"]
# #     #
# #     # For each room, count how many "expected" objects are present.
# #     for room, keywords in ROOM_OBJECT_MAP.items():
# #         score = len(detected & set(keywords))  # intersection match count

# #         # Track the highest-scoring room based on object evidence.
# #         if score > best_score:
# #             best_score = score
# #             best_label = room

# #     # IMPORTANT RULE:
# #     # Do NOT override the classifier unless >= 2 matching objects are found.
# #     # This avoids making wild jumps because a single object appeared.
# #     return best_label if best_score >= 2 else raw_label


# # ---------------------------------------------------------
# # BUILD RULE-BASED SUMMARY
# # ---------------------------------------------------------
# def _build_rule_summary(label: str, scene: SceneContextResult, det_result, llm_text: str) -> str:
#     obj_counts, total = _extract_object_counts(det_result)
#     top_objs = [o for o, _ in obj_counts.most_common(2)]

#     obj_fragment = ""
#     if total > 0:
#         obj_fragment = f" with {', '.join(top_objs)}" if top_objs else " with detected objects"

#     txt = (
#         f"{label.capitalize()}{obj_fragment}, "
#         f"crowd {scene.crowd_level}, "
#         f"risk {scene.risk_score:.2f}, "
#         f"{scene.navigation_hint}."
#     )

#     if len(txt) > 160:
#         txt = txt[:157] + "..."
#     return txt


# # ---------------------------------------------------------
# # DRAW OUTPUT
# # ---------------------------------------------------------
# def draw_predictions(
#     frame,
#     det_result,
#     scene_label: str,
#     conf: float,
#     scene: SceneContextResult,
#     summary_sentence: Optional[str],
# ):
#     """
#     Draw:
#         - YOLO bounding boxes
#         - Indoor classification label
#         - Crowd + Risk
#         - Navigation hint
#         - Summary sentence
#     """

#     img_h, img_w = frame.shape[:2]

#     # =========================================================
#     # YOLO BOUNDING BOXES  (Green)
#     # =========================================================
#     if det_result is not None and det_result.boxes is not None:
#         for box, cls_id, conf_box in zip(
#             det_result.boxes.xyxy,
#             det_result.boxes.cls,
#             det_result.boxes.conf,
#         ):
#             x1, y1, x2, y2 = map(int, box.tolist())
#             label = det_result.names[int(cls_id)]
#             conf_val = float(conf_box)

#             # Box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

#             # Label
#             label_text = f"{label} {conf_val:.2f}"
#             (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#             cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)

#             cv2.putText(
#                 frame,
#                 label_text,
#                 (x1 + 5, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 (0, 0, 0),
#                 2,
#             )

#     # =========================================================
#     # SCENE LABEL — BIG RED
#     # =========================================================
#     cv2.putText(
#         frame,
#         f"{scene_label.upper()} ({conf:.2f})",
#         (20, 60),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         2.0,
#         (0, 0, 255),  # STRONG RED
#         5,
#     )

#     # =========================================================
#     # CROWD + RISK — SOFT RED
#     # =========================================================
#     cv2.putText(
#         frame,
#         f"Crowd: {scene.crowd_level} | Risk: {scene.risk_score:.2f}",
#         (20, 120),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.3,
#         (0, 0, 220),  # SOFT RED
#         3,
#     )

#     # =========================================================
#     # NAVIGATION HINT — ORANGE
#     # =========================================================
#     cv2.putText(
#         frame,
#         f"Hint: {scene.navigation_hint}",
#         (20, 170),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.0,
#         (50, 50, 255),  # ORANGE
#         3,
#     )

#     # =========================================================
#     # SUMMARY — MAGENTA
#     # =========================================================
#     if summary_sentence:
#         cv2.putText(
#             frame,
#             f"Summary: {summary_sentence}",
#             (20, 230),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (180, 0, 180),  # MAGENTA
#             3,
#         )

#     return frame


# # ---------------------------------------------------------
# # LOAD MODELS
# # ---------------------------------------------------------
# def load_models():
#     detector = YOLO(str(paths["models"] / "yolo_detector" / "best.pt"))

#     # [CHANGED] IndoorClassifier now explicitly uses the ResNet Places365 / MIT checkpoint
#     #           instead of the old YOLOv8-CLS best.pt. The IndoorClassifier class
#     #           knows how to load and run resnet_places365_best.pth.
#     classifier = IndoorClassifier(
#         model_path=paths["models"] / "indoor_classification" / "resnet_places365_best.pth",
#         device="cpu",
#     )

#     return detector, classifier


# # ---------------------------------------------------------
# # IMAGE PROCESSING
# # ---------------------------------------------------------
# def process_image(path: Path, detector, classifier):
#     logger.info(f"[IMAGE] {path}")
#     img = cv2.imread(str(path))
#     if img is None:
#         logger.error("Image read failed.")
#         return

#     det = detector(img)[0]
#     cls = classifier.predict(img)

#     fused_label = refine_scene_with_objects(cls.label, det)
#     scene = reason_about_scene(fused_label, det)

#     llm_text = llm_reason_from_detections(fused_label, det, source=str(path))
#     summary = _build_rule_summary(fused_label, scene, det, llm_text)

#     frame = draw_predictions(img, det, fused_label, cls.confidence, scene, summary)

#     out = IMG_OUT / f"{path.stem}_full_{timestamp()}.jpg"
#     cv2.imwrite(str(out), frame)

#     logger.info(f"Saved → {out}")


# # ---------------------------------------------------------
# # VIDEO PROCESSING
# # ---------------------------------------------------------
# def process_video(path: Path, detector, classifier):
#     logger.info(f"[VIDEO] {path}")

#     cap = cv2.VideoCapture(str(path))
#     if not cap.isOpened():
#         logger.error("Failed to open video.")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS) or 24
#     w, h = int(cap.get(3)), int(cap.get(4))
#     out_path = VID_OUT / f"{path.stem}_full_{timestamp()}.mp4"
#     writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#     # First frame → LLM + summary
#     ret, first = cap.read()
#     if not ret:
#         return

#     det_first = detector(first)[0]
#     cls_first = classifier.predict(first)

#     fused_first = refine_scene_with_objects(cls_first.label, det_first)
#     scene_first = reason_about_scene(fused_first, det_first)

#     llm_text = llm_reason_from_detections(fused_first, det_first, source=str(path))
#     summary = _build_rule_summary(fused_first, scene_first, det_first, llm_text)

#     frame = draw_predictions(first, det_first, fused_first, cls_first.confidence, scene_first, summary)
#     writer.write(frame)

#     # Remaining frames (no LLM)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         det = detector(frame)[0]
#         cls = classifier.predict(frame)

#         fused = refine_scene_with_objects(cls.label, det)
#         scene = reason_about_scene(fused, det)

#         frame = draw_predictions(frame, det, fused, cls.confidence, scene, summary)
#         writer.write(frame)

#     cap.release()
#     writer.release()
#     logger.info(f"Saved video → {out_path}")


# # ---------------------------------------------------------
# # WEBCAM PROCESSING
# # ---------------------------------------------------------
# def process_webcam(detector, classifier, index: int = 0):
#     logger.info("[WEBCAM] Starting webcam")

#     cap = cv2.VideoCapture(index)
#     if not cap.isOpened():
#         logger.error("Webcam unavailable.")
#         return

#     fps = 30
#     w, h = int(cap.get(3)), int(cap.get(4))
#     out_path = VID_OUT / f"webcam_full_{timestamp()}.mp4"
#     writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#     ret, first = cap.read()
#     if not ret:
#         return

#     det_first = detector(first)[0]
#     cls_first = classifier.predict(first)

#     fused_first = refine_scene_with_objects(cls_first.label, det_first)
#     scene_first = reason_about_scene(fused_first, det_first)

#     llm_text = llm_reason_from_detections(
#         fused_first, det_first, source="webcam", print_to_console=False
#     )
#     summary = _build_rule_summary(fused_first, scene_first, det_first, llm_text)

#     frame = draw_predictions(first, det_first, fused_first, cls_first.confidence, scene_first, summary)
#     writer.write(frame)
#     cv2.imshow("RSAN Webcam", frame)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         det = detector(frame)[0]
#         cls = classifier.predict(frame)
#         fused = refine_scene_with_objects(cls.label, det)
#         scene = reason_about_scene(fused, det)

#         frame = draw_predictions(frame, det, fused, cls.confidence, scene, summary)
#         writer.write(frame)
#         cv2.imshow("RSAN Webcam", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()
#     logger.info(f"Saved webcam video → {out_path}")


# # ---------------------------------------------------------
# # MAIN ENTRY
# # ---------------------------------------------------------
# def run_unified(input_path: str):
#     detector, classifier = load_models()
#     p = Path(input_path)

#     if input_path.lower() in {"cam", "webcam"}:
#         return process_webcam(detector, classifier)

#     if p.is_file() and is_image(p):
#         return process_image(p, detector, classifier)

#     if p.is_file() and is_video(p):
#         return process_video(p, detector, classifier)

#     if p.is_dir():
#         for f in sorted(p.iterdir()):
#             if is_image(f):
#                 process_image(f, detector, classifier)
#             elif is_video(f):
#                 process_video(f, detector, classifier)
#         return

#     logger.error(f"Invalid input: {input_path}")


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) > 1:
#         run_unified(sys.argv[1])
#     else:
#         print("Usage: python -m src.tools.run_unified_pipeline <image|video|folder|webcam>")


# [CHANGED] Updated description to reflect ResNet Places365 indoor classifier (not YOLOv8-CLS).
"""
Unified RSAN Perception Pipeline (Hybrid + Safe LLM Version)
--------------------------------------------------------------------

Capabilities:
  • YOLOv8 object detection
  • Indoor scene classification (ResNet Places365 / MIT Indoor head)
  • Hybrid Fusion (YOLO objects refine indoor classification)
  • Scene-level reasoning (risk, crowd, hints)
  • Optional LLM reasoning (safe: disabled if no API key)
  • Clean 1-sentence summary overlay
  • Image, video, and webcam support

Team-Safe Behavior:
  • If OPENAI_API_KEY is missing → pipeline runs normally
  • LLM reasoning becomes fallback text
  • Red overlay: “LLM DISABLED — No API Key”
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
# PIPELINE PATHS
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
# SIMPLE UTILITIES
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
        data = json.load(open(JSON_LOG))
    else:
        data = []
    data.append(entry)
    json.dump(data, open(JSON_LOG, "w"), indent=4)


def _extract_object_counts(det_result) -> Tuple[Counter, int]:
    """Convert YOLO detection results → a count dict."""
    counts = Counter()
    total = 0

    if det_result is None or det_result.boxes is None:
        return counts, 0

    names = det_result.names
    cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)

    for cid in cls_ids:
        label = names.get(int(cid), f"id_{cid}")
        counts[label] += 1
        total += 1

    return counts, total


# ---------------------------------------------------------
# HYBRID FUSION (YOLO + Indoor CLS)
# ---------------------------------------------------------
def refine_scene_with_objects(raw_label: str, det_result, cls_confidence: float) -> str:
    """
    HYBRID FUSION (UPGRADED VERSION):
    Combines object evidence + classifier confidence using:

        scene_score = classifier_confidence * 0.6 + object_match_score * 0.4

    This gives smooth, stable, probabilistic refinement instead of the old
    "2-object rule".
    """

    if det_result is None or det_result.boxes is None:
        return raw_label

    names = det_result.names
    cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)
    detected = {names[int(cid)].lower() for cid in cls_ids}

    best_label = raw_label
    best_scene_score = cls_confidence

    for room, keywords in ROOM_OBJECT_MAP.items():

        matches = len(detected & set(keywords))
        object_match_score = matches / max(1, len(keywords))

        scene_score = cls_confidence * 0.6 + object_match_score * 0.4

        if scene_score > best_scene_score:
            best_scene_score = scene_score
            best_label = room

    return best_label


# ---------------------------------------------------------
# BUILD RULE-BASED SUMMARY
# ---------------------------------------------------------
def _build_rule_summary(label: str, scene: SceneContextResult, det_result, llm_text: str) -> str:
    obj_counts, total = _extract_object_counts(det_result)
    top_objs = [o for o, _ in obj_counts.most_common(2)]

    obj_fragment = ""
    if total > 0:
        obj_fragment = f" with {', '.join(top_objs)}" if top_objs else " with detected objects"

    txt = (
        f"{label.capitalize()}{obj_fragment}, "
        f"crowd {scene.crowd_level}, "
        f"risk {scene.risk_score:.2f}, "
        f"{scene.navigation_hint}."
    )

    if len(txt) > 160:
        txt = txt[:157] + "..."
    return txt


# ---------------------------------------------------------
# DRAW OUTPUT
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

    # YOLO BOXES
    if det_result is not None and det_result.boxes is not None:
        for box, cls_id, conf_box in zip(
            det_result.boxes.xyxy,
            det_result.boxes.cls,
            det_result.boxes.conf,
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = det_result.names[int(cls_id)]
            conf_val = float(conf_box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label_text = f"{label} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)

            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
            )

    # SCENE LABEL
    cv2.putText(
        frame,
        f"{scene_label.upper()} ({conf:.2f})",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        5,
    )

    # CROWD + RISK
    cv2.putText(
        frame,
        f"Crowd: {scene.crowd_level} | Risk: {scene.risk_score:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 220),
        3,
    )

    # NAVIGATION HINT
    cv2.putText(
        frame,
        f"Hint: {scene.navigation_hint}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (50, 50, 255),
        3,
    )

    # SUMMARY
    if summary_sentence:
        cv2.putText(
            frame,
            f"Summary: {summary_sentence}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 0, 180),
            3,
        )

    return frame


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
def load_models():
    detector = YOLO(str(paths["models"] / "yolo_detector" / "best.pt"))

    classifier = IndoorClassifier(
        model_path=paths["models"] / "indoor_classification" / "resnet_places365_best.pth",
        device="cpu",
    )

    return detector, classifier


# ---------------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------------
def process_image(path: Path, detector, classifier):
    logger.info(f"[IMAGE] {path}")
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Image read failed.")
        return

    det = detector(img)[0]
    cls = classifier.predict(img)

    fused_label = refine_scene_with_objects(cls.label, det, cls.confidence)
    scene = reason_about_scene(fused_label, det)

    llm_text = llm_reason_from_detections(fused_label, det, source=str(path))
    summary = _build_rule_summary(fused_label, scene, det, llm_text)

    frame = draw_predictions(img, det, fused_label, cls.confidence, scene, summary)

    out = IMG_OUT / f"{path.stem}_full_{timestamp()}.jpg"
    cv2.imwrite(str(out), frame)

    logger.info(f"Saved → {out}")


# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------
def process_video(path: Path, detector, classifier):
    logger.info(f"[VIDEO] {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error("Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(3)), int(cap.get(4))
    out_path = VID_OUT / f"{path.stem}_full_{timestamp()}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ret, first = cap.read()
    if not ret:
        return

    det_first = detector(first)[0]
    cls_first = classifier.predict(first)

    fused_first = refine_scene_with_objects(cls_first.label, det_first, cls_first.confidence)
    scene_first = reason_about_scene(fused_first, det_first)

    llm_text = llm_reason_from_detections(fused_first, det_first, source=str(path))
    summary = _build_rule_summary(fused_first, scene_first, det_first, llm_text)

    frame = draw_predictions(first, det_first, fused_first, cls_first.confidence, scene_first, summary)
    writer.write(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detector(frame)[0]
        cls = classifier.predict(frame)

        fused = refine_scene_with_objects(cls.label, det, cls.confidence)
        scene = reason_about_scene(fused, det)

        frame = draw_predictions(frame, det, fused, cls.confidence, scene, summary)
        writer.write(frame)

    cap.release()
    writer.release()
    logger.info(f"Saved video → {out_path}")


# ---------------------------------------------------------
# WEBCAM PROCESSING
# ---------------------------------------------------------
def process_webcam(detector, classifier, index: int = 0):
    logger.info("[WEBCAM] Starting webcam")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Webcam unavailable.")
        return

    fps = 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out_path = VID_OUT / f"webcam_full_{timestamp()}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ret, first = cap.read()
    if not ret:
        return

    det_first = detector(first)[0]
    cls_first = classifier.predict(first)

    fused_first = refine_scene_with_objects(cls_first.label, det_first, cls_first.confidence)
    scene_first = reason_about_scene(fused_first, det_first)

    llm_text = llm_reason_from_detections(fused_first, det_first, source="webcam", print_to_console=False)
    summary = _build_rule_summary(fused_first, scene_first, det_first, llm_text)

    frame = draw_predictions(first, det_first, fused_first, cls_first.confidence, scene_first, summary)
    writer.write(frame)
    cv2.imshow("RSAN Webcam", frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        det = detector(frame)[0]
        cls = classifier.predict(frame)
        fused = refine_scene_with_objects(cls.label, det, cls.confidence)
        scene = reason_about_scene(fused, det)

        frame = draw_predictions(frame, det, fused, cls.confidence, scene, summary)
        writer.write(frame)
        cv2.imshow("RSAN Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Saved webcam video → {out_path}")


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
def run_unified(input_path: str):
    detector, classifier = load_models()
    p = Path(input_path)

    if input_path.lower() in {"cam", "webcam"}:
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_unified(sys.argv[1])
    else:
        print("Usage: python -m src.tools.run_unified_pipeline <image|video|folder|webcam>")
