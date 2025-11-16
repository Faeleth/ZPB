from ultralytics import YOLO

if __name__ == "__main__":
    data_yaml = "../YOLO_format/data.yaml"

    # Train (or load trained model)
    model = YOLO("yolov8n.pt")
    # Continue training
    # model = YOLO("./results/yolov8n_training_epochs100/weights/best.pt")

    results = model.train(
        data=data_yaml,
        epochs=200,
        imgsz=800,
        batch=-1,  # auto
        name="yolov8n_training_epochs100",
        project="./results",
        device="0",
        mosaic=0.0,
    )
