import cv2
from ultralytics import YOLO
import numpy as np


class FaceRecognition:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Create a small blank image (128x128 black square)
        # This forces the GPU to initialize right now, not later.
        dummy_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        self.model.predict(dummy_frame, device="0", verbose=False)

    def predict(self, frame):
        # color format is expected to be BGR now
        # color format can be "BGR" or "RGB", but propably will be only BGR later
        w_max, h_max = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using OpenCV
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )

        plot_data = []

        for x, y, w, h in faces:
            extend_frame = 20
            x1 = int(max(0, x - extend_frame))
            y1 = int(max(0, y - extend_frame))
            x2 = int(min(w_max, x + w + extend_frame))
            y2 = int(min(h_max, y + h + extend_frame))

            # Sanity Check: Ensure the crop is valid before slicing
            # (Prevents crashes if x1 >= x2 due to weird coordinate inputs)
            if x2 <= x1 or y2 <= y1:
                print("Invalid crop dimensions, skipping...")
                continue

            # Crop the detected face area from the frame
            face_crop = frame[y1:y2, x1:x2]

            # YOLO expects images in RGB format, convert from BGR (OpenCV default)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Run YOLO prediction on the face crop
            results = self.model.predict(face_rgb, device="0", conf=0.5, verbose=False)

            # Extract predicted class and confidence
            if results and len(results[0].boxes) > 0:
                # Assuming single detection (one face/emotion)
                box = results[0].boxes[0]
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                # Get class label name
                emotion = self.model.names[cls_id]

                plot_data.append([emotion, conf])

                # Draw rectangle and label on original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} {conf:.2f}",
                    (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, plot_data
