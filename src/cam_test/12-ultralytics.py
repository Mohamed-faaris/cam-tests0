import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo26m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()