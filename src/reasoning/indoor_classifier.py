"""
Indoor Scene Classification using YOLOv8-CLS
-------------------------------------------
This module wraps the YOLOv8 classification model for use inside RSAN.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path


class IndoorSceneClassifier:
    """
    Classifier for indoor scenes (office, hallway, lab, classroom)
    using YOLOv8-CLS (CNN backbone).
    """

    def __init__(self, model_path="models/indoor_classification/best.pt"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load YOLOv8-cls model (CNN)
        self.model = YOLO(str(model_path))
        self.labels = self.model.names  # e.g., {0:'office',1:'hallway',...}

    def classify(self, image):
        """
        Classifies a scene from an image path or BGR image.
        Returns a dict with scene label, confidence, and probabilities.
        """

        # Load image if path
        if isinstance(image, str) or isinstance(image, Path):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image

        # Run YOLOv8-CLS inference
        result = self.model(img)[0]

        top_id = result.probs.top1
        label = self.labels[top_id]
        confidence = float(result.probs.top1conf)

        # Convert all probabilities to dictionary
        prob_dict = {self.labels[i]: float(result.probs.data[i]) for i in range(len(self.labels))}

        return {"scene": label, "confidence": confidence, "all_scores": prob_dict}
