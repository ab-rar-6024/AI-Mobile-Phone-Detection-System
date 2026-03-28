import cv2
from ultralytics import YOLO
import datetime
import threading
import winsound
import os

# 🔥 Better model
model = YOLO("yolov8l.pt")

cap = cv2.VideoCapture(0)

# 📁 Folder to save images
SAVE_FOLDER = "detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def play_alert():
    winsound.Beep(2000, 800)

# 🧠 Detection smoothing
detection_count = 0
DETECTION_THRESHOLD = 5

# 📸 Prevent too many images
last_capture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 720))

    results = model(
        frame,
        conf=0.4,
        imgsz=1280,
        classes=[67],
        verbose=False
    )

    phone_detected = False

    for r in results:
        for box in r.boxes:
            phone_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 🧠 Smooth detection
    if phone_detected:
        detection_count += 1
    else:
        detection_count = 0

    # 🚨 Confirmed detection
    if detection_count >= DETECTION_THRESHOLD:
        current_time = datetime.datetime.now().timestamp()

        # 🔊 Alert sound
        threading.Thread(target=play_alert, daemon=True).start()

        # 📝 Log
        with open("logs.txt", "a") as f:
            f.write(f"Phone detected at {datetime.datetime.now()}\n")

        # 📸 Capture image every 5 seconds
        if current_time - last_capture_time > 5:
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            filepath = os.path.join(SAVE_FOLDER, filename)

            cv2.imwrite(filepath, frame)
            print(f"📸 Image saved: {filepath}")

            last_capture_time = current_time

        # 🚨 Alert message
        cv2.putText(frame, "🚨 PHONE DETECTED 🚨",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 255),
                    4)

    cv2.imshow("Mobile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()