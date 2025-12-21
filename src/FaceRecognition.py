import cv2
from ultralytics import YOLO
import numpy as np


class FaceRecognition:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Load the Face Detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Warmup GPU
        dummy_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        self.model.predict(dummy_frame, device="0", verbose=False)

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )

        plot_data = []

        # If no faces, return early to save time
        if len(faces) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, plot_data

        # Instead of predicting inside the loop, we gather all faces first.
        face_crops = []
        face_coords = []

        h_max, w_max = frame.shape[:2]

        for x, y, w, h in faces:
            extend = 20
            x1 = int(max(0, x - extend))
            y1 = int(max(0, y - extend))
            x2 = int(min(w_max, x + w + extend))
            y2 = int(min(h_max, y + h + extend))

            face_crop = frame[y1:y2, x1:x2]

            # Convert to RGB for YOLO
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            face_crops.append(face_rgb)
            face_coords.append((x1, y1, x2, y2))

        # We send the LIST of images. YOLO processes them in parallel.
        if face_crops:
            results = self.model.predict(
                face_crops, device="0", conf=0.5, verbose=False
            )

            # Map results back to the coordinates
            for i, result in enumerate(results):
                box = result.boxes
                if len(box) > 0:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    emotion = self.model.names[cls_id]

                    plot_data.append([emotion, conf])

                    # Retrieve original coordinates
                    x1, y1, x2, y2 = face_coords[i]

                    # Draw on the original frame
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
