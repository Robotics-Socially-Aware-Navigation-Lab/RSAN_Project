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
# ---------------------------------------------------------------
# PURPOSE:
#   This list contains the names of the indoor room types that the
#   MIT head of our classifier was trained to recognize.
#
#   Each item in this list corresponds to one output neuron in the
#   MIT classification head (the "mit_head" in PlacesMITMultiHead).
#
# WHY ORDER MATTERS:
#   During training, the model learned the classes in THIS exact
#   order. The output index (0–10) coming from the MIT head must
#   match the correct name in this list. If the order changes,
#   predictions will map to the WRONG labels.
#
# WHERE IT IS USED:
#   • In IndoorClassifier.predict() → converting the MIT index
#     (argmax) into a human-readable room label.
#
#       Example:
#         mit_best_idx = 2
#         label = MIT_CLASS_NAMES[2] → "classroom"
#
#   • In the probability dictionary returned to the pipeline.
#
#   • Used by downstream components (scene context, LLM reasoning,
#     HUD text) to understand and describe the predicted room.
#
# IMPORTANT:
#   Do NOT add new labels unless you retrain the model. The MIT head
#   only outputs 11 logits because it was trained on 11 classes.
#   Adding names here without retraining will break the classifier.
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

ALLOWED_PLACES_CLASSES = [  # this order does not matter, just make sure it exists in places365 (github)
    "office",
    "corridor",
    "classroom",
    "kitchen",
    "bathroom",
    "library",
    "living_room",
    "dining_room",
    "computer_room",
    "cafeteria",
    "lobby",
    "auditorium",
    "banquet_hall",
    "library/indoor",
    "bedroom",
    "church/indoor",
    "conference_room",
    "dining_hall",
    "garage/indoor",
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

    # This line inserts the trained weights into the architecture.
    # Before this, the model knows nothing.

    # The .pth file contains trained values for:
    # The ResNet backbone
    # The Places365 head
    # The MIT indoor head
    # These weights come from your training notebook.

    def __init__(self, num_mit_classes: int):
        super().__init__()

        # Load ResNet50 with 365 output neurons (Places-style)
        self.backbone = models.resnet50(num_classes=365)
        in_features = self.backbone.fc.in_features

        # Remove the final classification layer to get feature vectors
        self.backbone.fc = nn.Identity()

        # Places365 classification head (365 classes)
        self.places_head = nn.Linear(in_features, 365)

        # MIT classification head (The 11 indoor classes)
        self.mit_head = nn.Linear(in_features, num_mit_classes)

    def forward(self, x):
        # Pass input through ResNet backbone → get feature vector
        feats = self.backbone(x)

        # Run the features through BOTH heads
        places_logits = self.places_head(feats)  # size: [batch, 365]
        mit_logits = self.mit_head(feats)  # size: [batch, 11]

        # Return BOTH outputs
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
    This class is our indoor scene classifier.
    It combines two prediction systems:
        • MIT Indoor (11 indoor rooms)
        • Places365 (365 general scenes)
    It loads the trained model (.pth file) and the list of scene labels.
    It provides one main function: .predict(frame)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        labels_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        warmup: bool = True,
    ) -> None:

        # ------------------------------------------------------------
        # Load the file paths from the config (where the .pth and labels are)
        # ------------------------------------------------------------
        try:
            paths = load_paths()
        except Exception as exc:
            log.warning("Could not load paths; using default locations. %s", exc)
            paths = {}

        project_root = Path(__file__).resolve().parents[2]

        # If no model path given → use default resnet_places365_best.pth
        if model_path is None:
            model_path = paths.get(
                "indoor_classifier_model",
                project_root / "models" / "indoor_classification" / "resnet_places365_best.pth",
            )

        # If no label file given → use default Places365 label file
        if labels_path is None:
            labels_path = paths.get(
                "places365_labels",
                project_root / "models" / "indoor_classification" / "categories_places365.txt",
            )

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        # Make sure the .pth model file actually exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Indoor classifier model not found:\n  {self.model_path}\n"
                "Fix the file path in project_paths.yaml or put the model here."
            )

        # ------------------------------------------------------------
        # Choose device (GPU if available, otherwise CPU)
        # ------------------------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # If GPU requested but not available → use CPU
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available; switching to CPU.")
            self.device = "cpu"

        # ------------------------------------------------------------
        # Load the 365 Places365 class names from the text file
        # ------------------------------------------------------------
        # These names will match the indexes the model predicts.
        self.places_class_names: List[str] = _load_places365_labels(self.labels_path)

        if len(self.places_class_names) != 365:
            log.warning(
                "Expected 365 Places365 labels, got %d",
                len(self.places_class_names),
            )

        log.info("[IndoorClassifier] Loaded %d Places365 class names.", len(self.places_class_names))

        # ------------------------------------------------------------
        # Build a list of indices for the allowed Places365 classes
        # ------------------------------------------------------------
        # We only want to consider a subset of the 365 classes.
        # The names in ALLOWED_PLACES_CLASSES must exactly match
        # the entries in self.places_class_names.
        self.allowed_places_indices: List[int] = [
            i for i, name in enumerate(self.places_class_names) if name in ALLOWED_PLACES_CLASSES
        ]

        if not self.allowed_places_indices:
            log.warning(
                "[IndoorClassifier] No allowed Places365 classes matched. " "Falling back to using ALL 365 classes."
            )
            self.allowed_places_indices = list(range(len(self.places_class_names)))

        # ------------------------------------------------------------
        # Build the neural network + load the trained weights (.pth file)
        # ------------------------------------------------------------
        # 1. Create the model architecture (empty model, random weights)
        self.model = PlacesMITMultiHead(num_mit_classes=NUM_MIT_CLASSES)

        # 2. Load the learned weights from the .pth file
        #    This gives the model its "knowledge" to classify rooms.
        state = torch.load(str(self.model_path), map_location=self.device)

        # 3. Insert the trained weights into the model
        self.model.load_state_dict(state, strict=True)

        # Put the model on the selected device (CPU or GPU)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (no training)

        # ------------------------------------------------------------
        # Define how images are preprocessed before entering the model
        # (resize, crop, convert to tensor, normalize)
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

        # ------------------------------------------------------------
        # Warm-up: run one dummy prediction so the model is ready instantly
        # ------------------------------------------------------------
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
        This function looks at an image and decides:
            → What room or environment the robot is currently in.

        It uses ONE neural network with TWO "heads":
            1. MIT head: predicts one of your 11 indoor room types.
            2. Places365 head: predicts one of 365 general scene typ

        We compare the two predictions and choose the best one.
        This is called HYBRID FUSION.

        Return value:
            - The final room name (label)
            - The confidence score
            - All MIT class probabilities
            - Extra debug info
        """

        # Make sure we received a real image
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a valid numpy array")
        if frame.size == 0:
            raise ValueError("Empty frame passed to IndoorClassifier")

        # ------------------------------------------------------------------
        # STEP 1: Turn the image into the format the model expects.
        # ------------------------------------------------------------------
        # OpenCV uses BGR color format. We convert it to RGB.
        img = Image.fromarray(frame[:, :, ::-1])

        # Apply the same resizing and normalization used during training
        x = self.transform(img).unsqueeze(0).to(self.device)

        # ------------------------------------------------------------------
        # STEP 2: Run the model (get predictions from both heads)
        # ------------------------------------------------------------------
        with torch.no_grad():

            # Do the call to self.model here #
            places_logits, mit_logits = self.model(x)

            # Convert the outputs into probabilities (0–1)
            places_probs = F.softmax(places_logits, dim=1)[0]
            mit_probs = F.softmax(mit_logits, dim=1)[0]

        # ------------------------------------------------------------------
        # STEP 3: Find the BEST label from each head
        # ------------------------------------------------------------------

        # ---------------- MIT head (your 11 indoor classes) ----------------
        # Pick the MIT label with the highest probability.
        mit_best_idx = int(torch.argmax(mit_probs))
        mit_best_label = MIT_CLASS_NAMES[mit_best_idx]
        mit_best_conf = float(mit_probs[mit_best_idx])

        # ---------------- Places365 head (365 environment classes) --------
        # Pick the Places label with the highest probability,
        # but only among ALLOWED_PLACES_CLASSES.
        if self.allowed_places_indices:
            # Only consider allowed classes
            subset_probs = places_probs[self.allowed_places_indices]
            best_local_idx = int(torch.argmax(subset_probs))
            places_best_idx = self.allowed_places_indices[best_local_idx]
        else:
            # Fallback – should not happen in normal use
            places_best_idx = int(torch.argmax(places_probs))

        places_best_label = self.places_class_names[places_best_idx]
        places_best_conf = float(places_probs[places_best_idx])

        # ------------------------------------------------------------------
        # STEP 4: HYBRID FUSION
        # Decide which label we will trust as the final answer.
        # ------------------------------------------------------------------

        MIT_PRIORITY_THRESHOLD = 0.60  # MIT must be at least 60% confident

        # Special rule:
        # Places365 knows "cafeteria" but MIT does not.
        # If Places365 says "cafeteria", we accept it immediately.
        if places_best_label == "cafeteria":
            final_label = places_best_label
            final_conf = places_best_conf
            source = "Places365"

        # If MIT is confident (>= 60%), we trust MIT.
        elif mit_best_conf >= MIT_PRIORITY_THRESHOLD:
            final_label = mit_best_label
            final_conf = mit_best_conf
            source = "MIT"

        # Otherwise, MIT is unsure → use the Places365 label.
        else:
            final_label = places_best_label
            final_conf = places_best_conf
            source = "Places365"

        # ------------------------------------------------------------------
        # STEP 5: Build the output structure
        # ------------------------------------------------------------------

        # Create a dictionary of MIT class probabilities
        # (this is useful for debugging and charts)
        mit_prob_dict: Dict[str, float] = {MIT_CLASS_NAMES[i]: float(mit_probs[i]) for i in range(NUM_MIT_CLASSES)}

        # Extra debug info: shows which model we trusted and why
        raw_info = {
            "source": source,
            "mit_best": {"label": mit_best_label, "conf": mit_best_conf},
            "places_best": {"label": places_best_label, "conf": places_best_conf},
        }

        # Return everything to the rest of the system
        return IndoorClassificationResult(
            label=final_label,  # The final room/environment name
            confidence=final_conf,  # How confident the model is
            probs=mit_prob_dict,  # MIT probabilities (11 values)
            raw=raw_info,  # Debug info (useful for logs)
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
