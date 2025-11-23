from ultralytics import YOLO

if __name__ == "__main__":
    data_yaml = "../YOLO_format/data.yaml"

    # Train (or load trained model)
    model = YOLO("yolo11s.pt")
    # Continue training
    # model = YOLO("./results/yolo11s_training_epochs200/weights/best.pt")

    results = model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        batch=-1,  # auto
        name="yolo11s_training_epochs200",
        project="./results",
        device="0",
        mosaic=0.0,
    )
