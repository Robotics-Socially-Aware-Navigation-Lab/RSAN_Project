"""
Indoor classifier wrapper for RSAN Project.

This module loads the YOLOv8-CLS indoor scene classification model
(from models/indoor_classification/best.pt) and exposes a clean API
for predicting the room type for a given image frame.

Designed to be:
- Robust to missing models / bad frames
- Device-aware (GPU if available, else CPU)
- Easy to test and integrate into ROS/pipelines
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from ultralytics import YOLO

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class IndoorClassificationResult:
    """Structured result returned by IndoorClassifier."""
    label: str
    confidence: float
    probs: Dict[str, float]
    raw: Any  # underlying ultralytics Result object


class IndoorClassifier:
    """
    High-level wrapper around a YOLOv8-CLS indoor classification model.

    - Automatically locates the model file at:
          models/indoor_classification/best.pt
      relative to the project root.
    - Automatically selects GPU if available, otherwise CPU.
    - Provides a simple .predict(frame) API for BGR numpy frames.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        device: Optional[str] = None,
        warmup: bool = True,
    ) -> None:
        # Resolve project root as repo root (two levels above this file)
        self._root = Path(__file__).resolve().parents[2]

        # Resolve model path
        if model_path is None:
            model_path = self._root / "models" / "indoor_classification" / "best.pt"
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Indoor classification model not found at: {self.model_path}\n"
                "Make sure best.pt is saved under models/indoor_classification/best.pt"
            )

        # Device selection
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        log.info(
            "Initializing IndoorClassifier with model=%s on device=%s",
            self.model_path,
            self.device,
        )

        # Load YOLOv8-CLS model
        self.model = YOLO(str(self.model_path))

        # Optional warm-up to avoid first-frame latency spikes
        if warmup:
            try:
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                _ = self.model(dummy, device=self.device, verbose=False)
                log.info("IndoorClassifier warm-up completed.")
            except Exception as exc:
                log.warning("Warm-up failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, frame: np.ndarray) -> IndoorClassificationResult:
        """
        Predict the room category for a given frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image, e.g. from cv2.VideoCapture.read().

        Returns
        -------
        IndoorClassificationResult

        Raises
        ------
        ValueError if frame is invalid.
        RuntimeError if model inference fails.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a non-empty numpy array")
        if frame.size == 0:
            raise ValueError("Empty frame passed to IndoorClassifier.predict()")

        try:
            results = self.model(frame, device=self.device, verbose=False)[0]
        except Exception as exc:
            log.error("IndoorClassifier inference failed: %s", exc, exc_info=True)
            raise RuntimeError(f"IndoorClassifier inference failed: {exc}") from exc

        # Top-1 prediction
        top_idx = int(results.probs.top1)
        label = results.names[top_idx]
        confidence = float(results.probs.top1conf)

        # Full probability distribution
        probs = results.probs.data.float().cpu().numpy().tolist()
        prob_dict = {results.names[i]: float(p) for i, p in enumerate(probs)}

        return IndoorClassificationResult(
            label=label,
            confidence=confidence,
            probs=prob_dict,
            raw=results,
        )


# Singleton convenience
_classifier_singleton: Optional[IndoorClassifier] = None


def get_indoor_classifier() -> IndoorClassifier:
    """Get a singleton instance of IndoorClassifier (lazy-loaded)."""
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = IndoorClassifier()
    return _classifier_singleton


def classify_room(frame: np.ndarray) -> IndoorClassificationResult:
    """
    Shortcut: classify a frame using the global IndoorClassifier singleton.
    """
    clf = get_indoor_classifier()
    return clf.predict(frame)