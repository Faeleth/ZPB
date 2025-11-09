import cv2
from ultralytics import YOLO

# Load your trained YOLO emotion detection model
model = YOLO("emotion_yolo_model.pt")  # replace with your model path

# Load OpenCV Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        # Crop the detected face area from the frame
        face_crop = frame[y : y + h, x : x + w]

        # YOLO expects images in RGB format, convert from BGR (OpenCV default)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Run YOLO prediction on the face crop
        results = model.predict(
            face_rgb, imgsz=96, device="0"
        )  # adjust imgsz to model input size

        # Extract predicted class and confidence
        if results and len(results[0].boxes) > 0:
            # Assuming single detection (one face/emotion)
            box = results[0].boxes[0]
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()

            # Get class label name
            emotion = model.names[cls_id]

            # Draw rectangle and label on original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion} {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
