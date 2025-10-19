import os
import glob
import yaml


def load_data_yaml(data_dir):
    """
    Loads the data.yaml file from the dataset's root directory.
    This file contains class names and path information.

    Args:
        data_dir (str): The path to the unzipped dataset root folder
                        (e.g., './YOLO_format').

    Returns:
        dict: The parsed content of the data.yaml file, or None if not found.
    """
    yaml_path = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: data.yaml not found in {data_dir}")
        return None

    with open(yaml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as e:
            print(f"Error parsing data.yaml: {e}")
            return None


def load_yolo_split(data_dir, split="train", verbose=False):
    """
    Loads a specific split (train, valid, or test) of the YOLO dataset.

    The dataset structure is expected to be:
    data_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

    Args:
        data_dir (str): The path to the unzipped dataset root folder.
        split (str): The dataset split to load ('train', 'valid', or 'test').

    Returns:
        list: A list of dictionaries. Each dictionary contains:
              - 'image_path' (str): The full path to the image.
              - 'annotations' (list): A list of tuples, where each tuple is
                (class_id, center_x, center_y, width, height).
    """
    if verbose:
        print(f"\nLoading {split} data...")

    image_dir = os.path.join(data_dir, split, "images")
    label_dir = os.path.join(data_dir, split, "labels")

    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return []

    # Find all image files (png)
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith((".png"))]

    dataset = []

    for img_path in image_paths:
        # Get the corresponding label file path
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(label_dir, base_name + ".txt")

        annotations = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    try:
                        # YOLO format: class_id center_x center_y width height
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            annotations.append((class_id, cx, cy, w, h))
                    except ValueError:
                        print(
                            f"Warning: Skipping malformed line in {label_path}: {line.strip()}"
                        )

        dataset.append({"image_path": img_path, "annotations": annotations})

    if verbose:
        print(f"Found {len(dataset)} images in the {split} set.")

    return dataset


def load_yolo_dataset(data_dir, verbose=False):
    """
    Loads the complete YOLO dataset (train, valid, test splits) from a root directory.

    This function acts as the main loader. It first parses the 'data.yaml'
    file to get class names and dataset info. It then calls a helper
    function ('load_yolo_split') to load the image paths and annotations
    for each of the 'train', 'valid', and 'test' splits.

    An optional 'verbose' flag can be set to True to print the loaded
    class names and a detailed example of the first item in the training set.

    Args:
        data_dir (str): The path to the root directory of the YOLO dataset
                        (e.g., './YOLO_format'). This directory should
                        contain 'data.yaml' and the 'train', 'valid',
                        and 'test' subfolders.
        verbose (bool, optional): If True, prints detailed loading
                                  information. Defaults to False.

    Returns:
        tuple: A tuple containing four elements:
            - data_info (dict): The parsed content from 'data.yaml'.
            - train_data (list): A list of dicts for the training set.
            - valid_data (list): A list of dicts for the validation set.
            - test_data (list): A list of dicts for the test set.

            Each item in the data lists is a dictionary:
            {'image_path': str, 'annotations': list[tuple(int, float, float, float, float)]}
    """

    # 1. Load the data.yaml file to get class names
    data_info = load_data_yaml(data_dir)
    class_names = []
    if data_info and "names" in data_info:
        class_names = data_info["names"]
        if verbose:
            # class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            print(f"Successfully loaded {len(class_names)} classes: {class_names}")
    else:
        print("Could not load class names from data.yaml")

    # 2. Load the data
    train_data = load_yolo_split(data_dir, split="train")
    valid_data = load_yolo_split(data_dir, split="valid")
    test_data = load_yolo_split(data_dir, split="test")

    # 3. Inspect the first item in the training data
    if verbose and train_data:
        first_item = train_data[0]
        print(first_item)
        print(f"\n--- Example: First Training Item ---")
        print(f"Image Path: {first_item['image_path']}")
        print(f"Annotations (count): {len(first_item['annotations'])}")

        if first_item["annotations"]:
            # Get the first annotation
            first_annotation = first_item["annotations"][0]
            class_id, cx, cy, w, h = first_annotation

            # Map class_id to class name
            label_name = class_names[class_id] if class_names else "Unknown"

            print(f"  Annotation 1:")
            print(f"    Class ID:   {class_id} ({label_name})")
            print(f"    Center (x,y): ({cx}, {cy})")
            print(f"    Size (w,h):   ({w}, {h})")

    return (data_info, train_data, valid_data, test_data)
