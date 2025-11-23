"""
detect_utils.py

High-level detection API for RSAN.
Other modules (scene reasoning, ROS node, etc.) call THIS file.
"""

from __future__ import annotations
import logging
from typing import List
import numpy as np

from src.perception.detect_objects import (
    get_default_detector,
    Detection,
)


logger = logging.getLogger(__name__)


def run_detection(
    frame_bgr: np.ndarray,
    conf_threshold: float | None = None,
) -> List[Detection]:
    """
    High-level API for object detection.

    Args:
        frame_bgr: Input frame as BGR np.ndarray.
        conf_threshold: Override minimum detection confidence.

    Returns:
        List[Detection]: Structured detection objects.
    """

    if frame_bgr is None or frame_bgr.size == 0:
        logger.warning("run_detection received an empty frame.")
        return []

    detector = get_default_detector()

    try:
        detections = detector.detect(frame_bgr)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return []

    return detections


def run_and_print(frame_bgr: np.ndarray) -> None:
    """Debug helper: prints detections to console."""
    dets = run_detection(frame_bgr)
    for d in dets:
        print(f"{d.class_name}  {d.confidence:.2f}  {d.bbox_xyxy}")