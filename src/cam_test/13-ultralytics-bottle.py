from ultralytics import YOLO
import cv2

model = YOLO("yolo26l.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame, classes=[39])

    output = results[0].plot()

    cv2.imshow("Bottle Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()