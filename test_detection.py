import cv2

from src.perception.detect_utils import run_detection

IMAGE_PATH = "/Users/rolandoyax/Desktop/photo.jpg"

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"[ERROR] Could not load image: {IMAGE_PATH}")
    print("[HINT] Check that the image is still on your Desktop.")
    exit()

detections = run_detection(frame)

print("\n=== DETECTIONS ===")
if not detections:
    print("No objects detected.")
else:
    for det in detections:
        print(f"{det.class_name}: {det.confidence:.2f}  {det.bbox_xyxy}")
