from ultralytics import YOLO
import os

if __name__ == "__main__":
    # 1. Path to the LAST checkpoint from your interrupted run
    checkpoint_path = "./results/yolo11s_training_epochs200/weights/last.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found at: {checkpoint_path}")
    else:
        # 2. Load the model from that checkpoint
        model = YOLO(checkpoint_path)

        # 3. Call train() with resume=True
        # It will automatically find all other settings (data, imgsz, etc.)
        # and continue from epoch 40 all the way to 100.
        results = model.train(resume=True)

        print("Training resumed and completed!")
