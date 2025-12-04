# """
# Indoor classifier wrapper for RSAN Project.

# This module loads the YOLOv8-CLS indoor scene classification model
# (from models/indoor_classification/best.pt) and exposes a clean API
# for predicting the room type for a given image frame.

# Designed to be:
# - Robust to missing models / bad frames
# - Device-aware (GPU if available, else CPU)
# - Easy to test and integrate into ROS/pipelines
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, Optional

# import numpy as np
# import torch
# from ultralytics import YOLO

# from src.utils.logger import get_logger

# log = get_logger(__name__)


# @dataclass
# class IndoorClassificationResult:
#     """Structured result returned by IndoorClassifier."""

#     label: str
#     confidence: float
#     probs: Dict[str, float]
#     raw: Any  # underlying ultralytics Result object


# class IndoorClassifier:
#     """
#     High-level wrapper around a YOLOv8-CLS indoor classification model.

#     - Automatically locates the model file at:
#           models/indoor_classification/best.pt
#       relative to the project root.
#     - Automatically selects GPU if available, otherwise CPU.
#     - Provides a simple .predict(frame) API for BGR numpy frames.
#     """

#     def __init__(
#         self,
#         model_path: Optional[Path | str] = None,
#         device: Optional[str] = None,
#         warmup: bool = True,
#     ) -> None:
#         # Resolve project root as repo root (two levels above this file)
#         self._root = Path(__file__).resolve().parents[2]

#         # Resolve model path
#         if model_path is None:
#             model_path = self._root / "models" / "indoor_classification" / "best.pt"
#         self.model_path = Path(model_path)

#         if not self.model_path.exists():
#             raise FileNotFoundError(
#                 f"Indoor classification model not found at: {self.model_path}\n"
#                 "Make sure best.pt is saved under models/indoor_classification/best.pt"
#             )

#         # Device selection
#         if device is not None:
#             self.device = device
#         else:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         if self.device.startswith("cuda") and not torch.cuda.is_available():
#             log.warning("CUDA requested but not available. Falling back to CPU.")
#             self.device = "cpu"

#         log.info(
#             "Initializing IndoorClassifier with model=%s on device=%s",
#             self.model_path,
#             self.device,
#         )

#         # Load YOLOv8-CLS model
#         self.model = YOLO(str(self.model_path))

#         # Optional warm-up to avoid first-frame latency spikes
#         if warmup:
#             try:
#                 dummy = np.zeros((224, 224, 3), dtype=np.uint8)
#                 _ = self.model(dummy, device=self.device, verbose=False)
#                 log.info("IndoorClassifier warm-up completed.")
#             except Exception as exc:
#                 log.warning("Warm-up failed: %s", exc)

#     # ------------------------------------------------------------------ #
#     # Public API
#     # ------------------------------------------------------------------ #
#     def predict(self, frame: np.ndarray) -> IndoorClassificationResult:
#         """
#         Predict the room category for a given frame.

#         Parameters
#         ----------
#         frame : np.ndarray
#             BGR image, e.g. from cv2.VideoCapture.read().

#         Returns
#         -------
#         IndoorClassificationResult

#         Raises
#         ------
#         ValueError if frame is invalid.
#         RuntimeError if model inference fails.
#         """
#         if frame is None or not isinstance(frame, np.ndarray):
#             raise ValueError("Frame must be a non-empty numpy array")
#         if frame.size == 0:
#             raise ValueError("Empty frame passed to IndoorClassifier.predict()")

#         try:
#             results = self.model(frame, device=self.device, verbose=False)[0]
#         except Exception as exc:
#             log.error("IndoorClassifier inference failed: %s", exc, exc_info=True)
#             raise RuntimeError(f"IndoorClassifier inference failed: {exc}") from exc

#         # Top-1 prediction
#         top_idx = int(results.probs.top1)
#         label = results.names[top_idx]
#         confidence = float(results.probs.top1conf)

#         # Full probability distribution
#         probs = results.probs.data.float().cpu().numpy().tolist()
#         prob_dict = {results.names[i]: float(p) for i, p in enumerate(probs)}

#         return IndoorClassificationResult(
#             label=label,
#             confidence=confidence,
#             probs=prob_dict,
#             raw=results,
#         )


# # Singleton convenience
# _classifier_singleton: Optional[IndoorClassifier] = None


# def get_indoor_classifier() -> IndoorClassifier:
#     """Get a singleton instance of IndoorClassifier (lazy-loaded)."""
#     global _classifier_singleton
#     if _classifier_singleton is None:
#         _classifier_singleton = IndoorClassifier()
#     return _classifier_singleton


# def classify_room(frame: np.ndarray) -> IndoorClassificationResult:
#     """
#     Shortcut: classify a frame using the global IndoorClassifier singleton.
#     """
#     clf = get_indoor_classifier()
#     return clf.predict(frame)


"""
Indoor classifier wrapper for RSAN Project (MIT+Places365 multi-head).

Loads your fine-tuned multi-head model:
    resnet_places365_best.pth

This .pth contains:
    backbone.*          → ResNet50 features
    places_head.*       → 365 Places365 classes
    mit_head.*          → Your indoor classes (MIT classes)

We use ONLY the MIT head for indoor classification in the RSAN pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from src.utils.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------
# MIT class list — MUST match your fine-tuned model order
# --------------------------------------------------------------
MIT_CLASS_NAMES: List[str] = [
    "bathroom",
    "bedroom",
    "classroom",
    "colloquium",
    "common_area",
    "computer_lab",
    "hallway",
    "kitchen",
    "library",
    "living_room",
    "office",
]
NUM_MIT_CLASSES = len(MIT_CLASS_NAMES)


# --------------------------------------------------------------
# Return structure for RSAN
# --------------------------------------------------------------
@dataclass
class IndoorClassificationResult:
    label: str
    confidence: float
    probs: Dict[str, float]
    raw: Any  # raw logits


# --------------------------------------------------------------
# The EXACT multi-head architecture your .pth requires
# --------------------------------------------------------------
class PlacesMITMultiHead(nn.Module):
    """
    Multi-head ResNet50 model:

       backbone → ResNet50 (features only)
       places_head → 365 classes
       mit_head → NUM_MIT_CLASSES indoor classes

    This architecture matches the one used to train:
       resnet_places365_best.pth
    """

    def __init__(self, num_mit_classes: int):
        super().__init__()

        # Base ResNet50 (same as notebook)
        self.backbone = models.resnet50(num_classes=365)
        in_features = self.backbone.fc.in_features

        # Replace classifier with identity → output feature vector
        self.backbone.fc = nn.Identity()

        # Original Places365 head (365-way)
        self.places_head = nn.Linear(in_features, 365)

        # Fine-tuned MIT indoor head
        self.mit_head = nn.Linear(in_features, num_mit_classes)

    def forward(self, x):
        feats = self.backbone(x)
        places_logits = self.places_head(feats)
        mit_logits = self.mit_head(feats)
        return places_logits, mit_logits


# --------------------------------------------------------------
# MAIN CLASSIFIER WRAPPER USED BY RSAN
# --------------------------------------------------------------
class IndoorClassifier:
    """
    Loads the multi-head model resnet_places365_best.pth
    and exposes simple .predict(frame) interface.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        warmup: bool = True,
    ) -> None:

        # Resolve project root
        self._root = Path(__file__).resolve().parents[2]

        # DEFAULT MODEL PATH
        if model_path is None:
            model_path = self._root / "models" / "indoor_classification" / "resnet_places365_best.pth"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Indoor classifier model not found: {self.model_path}\n"
                "Expected: RSAN_Project/models/indoor_classification/resnet_places365_best.pth"
            )

        # Select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but not available — using CPU")
            self.device = "cpu"

        log.info(f"Loading IndoorClassifier model: {self.model_path} on device {self.device}")

        # Build architecture
        self.model = PlacesMITMultiHead(num_mit_classes=NUM_MIT_CLASSES)

        # Load fine-tuned weights
        state = torch.load(str(self.model_path), map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        log.info("IndoorClassifier ready.")

        # Transform (same as notebook)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if warmup:
            try:
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                _ = self.predict(dummy)
                log.info("IndoorClassifier warm-up completed.")
            except Exception as exc:
                log.warning(f"Warm-up failed: {exc}")

    # ----------------------------------------------------------
    # MAIN PREDICT METHOD (Used by Unified Pipeline)
    # ----------------------------------------------------------
    def predict(self, frame: np.ndarray) -> IndoorClassificationResult:

        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a non-empty numpy array")
        if frame.size == 0:
            raise ValueError("Empty frame received")

        # BGR → RGB → PIL image
        img = Image.fromarray(frame[:, :, ::-1])

        # Preprocess
        x = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            _, mit_logits = self.model(x)
            mit_probs = F.softmax(mit_logits, dim=1)[0]

        # Top prediction
        top_idx = int(torch.argmax(mit_probs))
        label = MIT_CLASS_NAMES[top_idx]
        confidence = float(mit_probs[top_idx])

        # Probability dict
        probs = {MIT_CLASS_NAMES[i]: float(mit_probs[i]) for i in range(NUM_MIT_CLASSES)}

        return IndoorClassificationResult(
            label=label,
            confidence=confidence,
            probs=probs,
            raw=mit_logits.cpu().numpy(),
        )


# --------------------------------------------------------------
# Singleton helpers (RSAN expects these)
# --------------------------------------------------------------
_classifier_singleton: Optional[IndoorClassifier] = None


def get_indoor_classifier() -> IndoorClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = IndoorClassifier()
    return _classifier_singleton


def classify_room(frame: np.ndarray) -> IndoorClassificationResult:
    clf = get_indoor_classifier()
    return clf.predict(frame)
