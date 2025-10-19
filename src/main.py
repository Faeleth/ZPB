import cv2
import numpy as np
import matplotlib.pyplot as plt

import database_loader as dl

DATASET_ROOT_DIR = ".\..\YOLO_format"
VERBOSE = True


if __name__ == "__main__":
    (data_info, train_data, valid_data, test_data) = dl.load_yolo_dataset(
        DATASET_ROOT_DIR, VERBOSE
    )
