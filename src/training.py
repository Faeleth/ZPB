from ultralytics import YOLO

if __name__ == "__main__":
    data_yaml = "..\YOLO_format\data.yaml"

    # Train (or load trained model)
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, epochs=100, device="0")

    # Save the trained model manually (optional)
    model.save("emotion_yolo_model.pt")

    # Later load and use the saved model
    model = YOLO("emotion_yolo_model.pt")
