"""
Scene Context Engine
--------------------
Combines:
- YOLO object detection
- Indoor scene classification (YOLOv8-CLS)
- LLM reasoning for social navigation
"""

from perception.detect_objects import detect_objects
from reasoning.indoor_classifier import IndoorSceneClassifier
from reasoning.llm_reasoner import LLMReasoner


class SceneContextEngine:
    def __init__(self):
        self.classifier = IndoorSceneClassifier()
        self.llm = LLMReasoner()

    def analyze(self, image_path):
        # Step 1 — YOLO object detection
        detections = detect_objects(image_path)

        # Step 2 — Scene classification (CNN)
        classification = self.classifier.classify(image_path)

        # Step 3 — Build context bundle
        context = {
            "scene": classification["scene"],
            "scene_confidence": classification["confidence"],
            "object_detections": detections,
            "probabilities": classification["all_scores"],
        }

        # Step 4 — LLM reasoning
        reasoning = self.llm.generate_reasoning(context)

        return {"classification": classification, "detections": detections, "reasoning": reasoning}
