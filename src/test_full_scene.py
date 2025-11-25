import cv2
from ultralytics import YOLO

from src.reasoning.indoor_classifier import IndoorClassifier
from src.reasoning.scene_context import reason_about_scene

IMAGE_PATH = "test.jpg"  # change this

det = YOLO("models/yolo_detector/best.pt")
clf = IndoorClassifier()

img = cv2.imread(IMAGE_PATH)
det_out = det(img)[0]

cls_out = clf.predict(img)
scene = reason_about_scene(cls_out.label, det_out)

print("=== Full Scene Understanding ===")
print("Room:", cls_out.label)
print("Confidence:", cls_out.confidence)
print("Crowd Level:", scene.crowd_level)
print("Risk Score:", scene.risk_score)
print("Navigation Hint:", scene.navigation_hint)
