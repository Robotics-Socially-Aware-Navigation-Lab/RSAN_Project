"""
Indoor classifier wrapper for RSAN Project (MIT + Places365 multi-head, hybrid fusion).

This classifier loads your fine-tuned model:
    resnet_places365_best.pth

The model contains:
    backbone.*      → ResNet50 feature extractor
    places_head.*   → 365 Places365 logits
    mit_head.*      → Your indoor MIT classes

RSAN pipeline relies on:
    • final_label (hybrid MIT + Places365)
    • confidence
    • probs over MIT classes (for compatibility)
    • raw dict with extra debug info (MIT vs Places365)

Config-driven paths:
    configs/project_paths.yaml → paths.indoor_classifier_model
                               → paths.places365_labels

Author: RSAN_Project
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from src.utils.file_utils import load_paths
from src.utils.logger import get_logger

log = get_logger(__name__)

# ================================================================
# MIT INDOOR CLASS LABELS
# MUST match EXACT order used during fine-tuning
# ================================================================
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


# ================================================================
# RSAN RETURN STRUCTURE (unchanged interface)
# ================================================================
@dataclass
class IndoorClassificationResult:
    label: str
    confidence: float
    probs: Dict[str, float]
    raw: Any  # extra info (e.g., logits, MIT vs Places365, source)


# ================================================================
# MULTI-HEAD RESNET ARCHITECTURE
# MUST MATCH TRAINING NOTEBOOK
# ================================================================
class PlacesMITMultiHead(nn.Module):
    """
    Multi-head ResNet50 architecture:

        backbone     → ResNet50 feature extractor
        places_head  → 365-way Places365 classifier
        mit_head     → fine-tuned MIT indoor classifier
    """

    def __init__(self, num_mit_classes: int):
        super().__init__()

        # ResNet50 configured for 365 Places classes
        self.backbone = models.resnet50(num_classes=365)
        in_features = self.backbone.fc.in_features

        # Replace final layer with Identity → feature vector
        self.backbone.fc = nn.Identity()

        # Places365 head (kept for hybrid fusion)
        self.places_head = nn.Linear(in_features, 365)

        # MIT indoor head (fine-tuned)
        self.mit_head = nn.Linear(in_features, num_mit_classes)

    def forward(self, x):
        feats = self.backbone(x)
        places_logits = self.places_head(feats)
        mit_logits = self.mit_head(feats)
        return places_logits, mit_logits


# ================================================================
# Helper: load Places365 label names from categories_places365.txt
# ================================================================
def _load_places365_labels(path: Path) -> List[str]:
    """
    Expected file format (standard Places365):

        /a/abbey 0
        /b/bowling_alley 1
        ...

    We keep only the human-readable name (last part after '/').
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Places365 label file not found at:\n  {path}\n" "Make sure categories_places365.txt is placed there."
        )

    num_classes = 365
    classes: List[Optional[str]] = [None] * num_classes

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            category_full = parts[0]  # e.g. '/a/abbey'
            try:
                cls_id = int(parts[-1])  # final token is the ID
            except ValueError:
                continue

            name = category_full.split("/")[-1]  # '/a/abbey' → 'abbey'

            if 0 <= cls_id < num_classes:
                classes[cls_id] = name

    # Fill any missing class names
    for i in range(num_classes):
        if classes[i] is None:
            classes[i] = f"class_{i}"

    return [str(c) for c in classes]


# ================================================================
# MAIN CLASSIFIER WRAPPER USED BY THE RSAN PIPELINE
# ================================================================
class IndoorClassifier:
    """
    Hybrid MIT + Places365 indoor scene classifier.

    Loads:
        - model weights from project_paths.yaml → paths.indoor_classifier_model
        - Places365 label file from paths.places365_labels

    Exposes:
        .predict(frame: np.ndarray) → IndoorClassificationResult
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        labels_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        warmup: bool = True,
    ) -> None:

        # ------------------------------------------------------------
        # Resolve paths via configs/project_paths.yaml
        # ------------------------------------------------------------
        try:
            paths = load_paths()
        except Exception as exc:
            log.warning("Failed to load project paths; using defaults. %s", exc)
            paths = {}

        project_root = Path(__file__).resolve().parents[2]

        if model_path is None:
            model_path = paths.get(
                "indoor_classifier_model",
                project_root / "models" / "indoor_classification" / "resnet_places365_best.pth",
            )

        if labels_path is None:
            labels_path = paths.get(
                "places365_labels",
                project_root / "models" / "indoor_classification" / "categories_places365.txt",
            )

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Indoor classifier model missing:\n  {self.model_path}\n"
                "Set paths.indoor_classifier_model in configs/project_paths.yaml "
                "or place the model at the default location."
            )

        # ------------------------------------------------------------
        # Device selection
        # ------------------------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but not available → falling back to CPU")
            self.device = "cpu"

        # ------------------------------------------------------------
        # Load Places365 labels
        # ------------------------------------------------------------
        self.places_class_names: List[str] = _load_places365_labels(self.labels_path)
        if len(self.places_class_names) != 365:
            log.warning(
                "Expected 365 Places365 labels, got %d",
                len(self.places_class_names),
            )

        log.info("[IndoorClassifier] Loaded %d Places365 class names.", len(self.places_class_names))

        # ------------------------------------------------------------
        # Build model + load weights
        # ------------------------------------------------------------
        log.info("[IndoorClassifier] Loading model weights from %s", self.model_path)

        self.model = PlacesMITMultiHead(num_mit_classes=NUM_MIT_CLASSES)

        state = torch.load(str(self.model_path), map_location=self.device)
        # strict=True because training script used same architecture
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # ------------------------------------------------------------
        # Input transform
        # ------------------------------------------------------------
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

        log.info("[IndoorClassifier] Ready on device=%s", self.device)

        # Optional warm-up for smoother first run
        if warmup:
            try:
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                _ = self.predict(dummy)
            except Exception as exc:
                log.warning("Warm-up failed: %s", exc)

    # ---------------------------------------------------------------
    # PREDICTION (Hybrid MIT + Places365)
    # ---------------------------------------------------------------
    def predict(self, frame: np.ndarray) -> IndoorClassificationResult:
        """
        Predict indoor scene label with hybrid fusion:

            - If MIT is confident → trust MIT label
            - Else → fall back to Places365 label
            - Special cases (e.g., cafeteria) can override behavior
        """

        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a valid numpy array")
        if frame.size == 0:
            raise ValueError("Empty frame passed to IndoorClassifier")

        # BGR → RGB and to PIL
        img = Image.fromarray(frame[:, :, ::-1])

        # Preprocess
        x = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            places_logits, mit_logits = self.model(x)
            places_probs = F.softmax(places_logits, dim=1)[0]
            mit_probs = F.softmax(mit_logits, dim=1)[0]

        # This where the classes name and confidance are slected **************
        # *********************************************************************
        # ---------------- MIT head (indoors, 11 classes) ----------------
        mit_best_idx = int(torch.argmax(mit_probs))
        mit_best_label = MIT_CLASS_NAMES[mit_best_idx]
        mit_best_conf = float(mit_probs[mit_best_idx])

        # ---------------- Places365 head (365 classes) ------------------
        places_best_idx = int(torch.argmax(places_probs))
        places_best_label = self.places_class_names[places_best_idx]
        places_best_conf = float(places_probs[places_best_idx])

        # ---------------------------------------------------------------
        # HYBRID FUSION RULES
        # ---------------------------------------------------------------
        # You can tune this threshold; keep 0.6 as in your Colab script.
        MIT_PRIORITY_THRESHOLD = 0.60

        # Example special case: cafeteria is often missed by MIT head
        if places_best_label == "cafeteria":
            final_label = places_best_label
            final_conf = places_best_conf
            source = "Places365"

        # If MIT is confident, trust MIT
        elif mit_best_conf >= MIT_PRIORITY_THRESHOLD:
            final_label = mit_best_label
            final_conf = mit_best_conf
            source = "MIT"

        # Otherwise, trust Places365
        else:
            final_label = places_best_label
            final_conf = places_best_conf
            source = "Places365"

        # ---------------------------------------------------------------
        # Build probability dict (MIT only, for backward compatibility)
        # ---------------------------------------------------------------
        mit_prob_dict: Dict[str, float] = {MIT_CLASS_NAMES[i]: float(mit_probs[i]) for i in range(NUM_MIT_CLASSES)}

        # `raw` can carry any extra debug info without breaking callers
        raw_info = {
            "source": source,
            "mit_best": {"label": mit_best_label, "conf": mit_best_conf},
            "places_best": {"label": places_best_label, "conf": places_best_conf},
        }

        return IndoorClassificationResult(
            label=final_label,
            confidence=final_conf,
            probs=mit_prob_dict,
            raw=raw_info,
        )


# ================================================================
# Singleton Interface (RSAN expects these names)
# ================================================================
_classifier_singleton: Optional[IndoorClassifier] = None


def get_indoor_classifier() -> IndoorClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = IndoorClassifier()
    return _classifier_singleton


def classify_room(frame: np.ndarray) -> IndoorClassificationResult:
    return get_indoor_classifier().predict(frame)
