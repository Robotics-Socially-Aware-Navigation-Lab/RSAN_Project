"""
Scene context reasoning for RSAN Project.

Combines:
    - room type from the indoor classifier
    - object detections from the YOLOv8 detector

to produce a high-level navigation recommendation suitable for
socially-aware navigation.

API is simple and future-proof for LLM-based reasoning later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SceneContextResult:
    room: str
    primary_objects: List[str]
    crowd_level: str
    risk_score: float
    navigation_hint: str


def _extract_object_names(detections: Any) -> List[str]:
    """
    Extract object class names from various formats.

    Supports:
    - ultralytics.yolo.engine.results.Results object
    - list[str] of class names
    - list[dict] with 'label' or 'class_name'
    - list[Results] (we take the first)
    """
    # Already list of strings
    if isinstance(detections, (list, tuple)) and all(isinstance(x, str) for x in detections):
        return list(detections)

    # List of dicts
    if isinstance(detections, (list, tuple)) and detections and all(isinstance(x, dict) for x in detections):
        names: List[str] = []
        for d in detections:
            if "label" in d:
                names.append(str(d["label"]))
            elif "class_name" in d:
                names.append(str(d["class_name"]))
        return names

    # Try ultralytics Results type
    try:
        from ultralytics.yolo.engine.results import Results  # type: ignore

        if isinstance(detections, list) and detections and isinstance(detections[0], Results):
            detections = detections[0]

        if isinstance(detections, Results):
            cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
            names_dict = detections.names
            return [names_dict[int(i)] for i in cls_ids]
    except Exception as exc:
        log.debug("Failed to parse detections as ultralytics Results: %s", exc)

    return []


def _estimate_crowd_level(object_names: Sequence[str]) -> str:
    """Simple heuristic for crowd density from number of people."""
    n_people = sum(1 for o in object_names if o == "person")
    if n_people == 0:
        return "empty"
    if n_people == 1:
        return "low"
    if 2 <= n_people <= 4:
        return "medium"
    return "high"


def _base_risk(room: str) -> float:
    """Assign a base risk level per room type."""
    room = room.lower()
    if room == "hallway":
        return 0.7  # narrow, dynamic
    if room == "lab":
        return 0.8  # fragile equipment
    if room == "classroom":
        return 0.6
    if room == "office":
        return 0.5
    return 0.5  # default for all future rooms


def _crowd_multiplier(crowd_level: str) -> float:
    return {
        "empty": 0.5,
        "low": 0.8,
        "medium": 1.0,
        "high": 1.2,
    }.get(crowd_level, 1.0)


# -----------------------------------------------------------
# SCALABLE NAVIGATION HINT FUNCTION
# -----------------------------------------------------------
def _compose_navigation_hint(room: str, object_names: Sequence[str], crowd_level: str) -> str:
    """
    Scalable, robust navigation-hint generator.
    Works with ANY number of rooms and ANY number of object classes.
    """

    # Normalize inputs
    room = (room or "unknown").lower()
    objs = set(o.lower() for o in object_names)
    n_people = sum(1 for o in objs if o == "person")

    # -------------------------------------------------------
    # 1. UNIVERSAL CROWD RULES
    # -------------------------------------------------------
    if n_people >= 3 or crowd_level == "high":
        return (
            "High crowd density detected. Reduce speed, maintain generous spacing, "
            "and wait for openings before proceeding."
        )

    if n_people == 2:
        return "Moderate crowd detected. Keep right, slow down, and avoid entering tight gaps."

    if n_people == 1:
        return "Single person detected. Maintain respectful distance and pass with smooth, predictable motion."

    # -------------------------------------------------------
    # 2. ROOM-SPECIFIC OPTIONAL LOGIC (safe for future scaling)
    # -------------------------------------------------------
    if room == "hallway":
        return (
            "Navigate along the right side for clear passage. Maintain predictable motion "
            "and watch for people approaching ahead."
        )

    if room == "classroom":
        if {"desk", "chair", "table"} & objs:
            return "Classroom with desks detected. Avoid cutting across rows; follow perimeter paths."
        return "Classroom detected. Move slowly and avoid sudden turns near seated individuals."

    if room == "lab":
        if {"bench", "monitor", "equipment", "bottle", "laptop", "keyboard"} & objs:
            return (
                "Lab environment detected. Maintain safe distance from benches and equipment; " "avoid narrow passages."
            )
        return "Lab detected. Proceed cautiously and maintain distance from fragile surfaces."

    if room == "office":
        if {"chair", "desk", "monitor", "computer"} & objs:
            return "Office detected. Navigate around desks and chairs, maintaining respectful distance."
        return "Office detected. Move at low speed and maintain predictable motion."

    # -------------------------------------------------------
    # 3. GENERIC RULES FOR UNKNOWN ROOMS
    # -------------------------------------------------------
    hazards = {
        "stove",
        "oven",
        "microwave",
        "sink",
        "knife",
        "glass",
        "refrigerator",
        "equipment",
        "cord",
        "cable",
        "ladder",
    }
    if objs & hazards:
        return "Potential hazards detected. Reduce speed and maintain extra distance from obstacles."

    furniture_keywords = {"desk", "table", "chair", "sofa", "couch"}
    if objs & furniture_keywords:
        return "Furniture-heavy environment. Navigate around large objects and avoid tight gaps."

    # Default fallback for ANY new future room types
    return (
        "Proceed cautiously, avoid close contact with obstacles, and maintain smooth, "
        "predictable motion through the environment."
    )


def reason_about_scene(room_label: str, detections: Any) -> SceneContextResult:
    """
    Main entry point: combine room + detections into a navigation recommendation.
    """
    object_names = _extract_object_names(detections)
    crowd_level = _estimate_crowd_level(object_names)

    risk = _base_risk(room_label) * _crowd_multiplier(crowd_level)
    risk = float(np.clip(risk, 0.0, 1.0))

    hint = _compose_navigation_hint(room_label, object_names, crowd_level)

    primary_objects = list(object_names[:10])

    return SceneContextResult(
        room=room_label,
        primary_objects=primary_objects,
        crowd_level=crowd_level,
        risk_score=risk,
        navigation_hint=hint,
    )
