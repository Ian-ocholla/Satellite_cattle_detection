"""
This code registers the COCO datasets and ensures all augmentations are applied.
"""

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import AugmentationList, ResizeShortestEdge, RandomFlip, RandomRotation, RandomBrightness, RandomContrast
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.utils.logger import setup_logger

logger = setup_logger()

# Set the working directory
WORK_DIR = "/projappl/project_2006327/Detectron/2025/satellite_dataset"  # Change this to your actual project directory
os.chdir(WORK_DIR)  # Change the working directory globally

def check_missing_files(json_file, image_root):
    """
    Checks if all images listed in the json_file exist in the image_root.
    """
    with open(json_file, 'r') as f:
        annotations = json.load(f)
        for img in annotations['images']:
            img_path = os.path.join(image_root, img['file_name'])
            if not os.path.exists(img_path):
                print(f"Missing file: {img_path}")

def register_datasets(name_ds):
    """
    Registers training, validation, and test datasets in COCO format.
    """
    name_ds_train = f"{name_ds}_train"
    name_ds_val = f"{name_ds}_val"
    name_ds_test = f"{name_ds}_test"

    image_root_train = os.path.join(name_ds, "train")
    image_root_val = os.path.join(name_ds, "val")
    image_root_test = os.path.join(name_ds, "test")

    af = "_annotations.coco.json"
    json_file_train = os.path.join(name_ds, "train", af)
    json_file_val = os.path.join(name_ds, "val", af)
    json_file_test = os.path.join(name_ds, "test", af)

    # Check for missing images
    check_missing_files(json_file_train, image_root_train)

    # Check for missing JSON files
    assert os.path.exists(json_file_train), f"JSON file not found: {json_file_train}"
    assert os.path.exists(json_file_val), f"JSON file not found: {json_file_val}"
    assert os.path.exists(json_file_test), f"JSON file not found: {json_file_test}"

    # Register datasets
    for name, json_file, image_root in zip(
        [name_ds_train, name_ds_val, name_ds_test],
        [json_file_train, json_file_val, json_file_test],
        [image_root_train, image_root_val, image_root_test],
    ):
        register_coco_instances(name=name, metadata={}, json_file=json_file, image_root=image_root)
        #MetadataCatalog.get(name).set(augments=build_augmentation())  # Apply augmentation

    logger.info(f"Datasets '{name_ds_train}', '{name_ds_val}', and '{name_ds_test}' registered successfully.")

    print(f"Training dataset name: {name_ds_train}")
    print(f"Validation dataset name: {name_ds_val}")
    print(f"Test dataset name: {name_ds_test}")

    return name_ds_train, name_ds_val, name_ds_test