import cv2
from ultralytics import YOLO
import datetime
import os

model = YOLO("yolov8l.pt")

SAVE_FOLDER = "static/detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(
            frame,
            conf=0.4,
            imgsz=960,
            classes=[0, 67],  # person + phone
            verbose=False
        )

        phone_detected = False
        person_box = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 👤 Person
                if cls == 0:
                    person_box = (x1, y1, x2, y2)

                # 📱 Phone
                if cls == 67:
                    phone_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 🚨 If both detected → capture
        if phone_detected and person_box:
            px1, py1, px2, py2 = person_box

            cropped = frame[py1:py2, px1:px2]

            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            filepath = os.path.join(SAVE_FOLDER, filename)

            cv2.imwrite(filepath, cropped)

            cv2.putText(frame, "CHEATING DETECTED",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3)

        # Convert to stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')