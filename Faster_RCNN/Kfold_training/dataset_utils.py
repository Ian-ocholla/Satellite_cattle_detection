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
KFOLD_BASE = "/projappl/project_2006327/Detectron/2025/Kfold_dataset/train_aug_data/coco_data/Kfolds"  # Change this to your actual project directory
assert os.path.exists(KFOLD_BASE), "Kfolds directory does not exist"

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

def register_all_kfolds(kfold_base):
    folds = sorted([d for d in os.listdir(kfold_base) if d.startswith("fold_")])
    fold_mapping={} 
    
    for fold_name in folds:
        fold_id = int(fold_name.split("_")[1])
        fold_path = os.path.join(kfold_base, fold_name)
        train_json = os.path.join(fold_path, "train_annotation.json")
        val_json = os.path.join(fold_path, "val_annotation.json")
        train_img_dir = os.path.join(fold_path, "images", "train")
        val_img_dir = os.path.join(fold_path, "images", "val")

        # Register dataset
        train_name = f"{fold_name}_train"
        val_name = f"{fold_name}_val"

        assert os.path.exists(train_json), f"Missing JSON: {train_json}"
        assert os.path.exists(val_json), f"Missing JSON: {val_json}"
        assert os.path.exists(train_img_dir), f"Missing images dir: {train_img_dir}"
        assert os.path.exists(val_img_dir), f"Missing images dir: {val_img_dir}"

        # Optional: check for missing images
        check_missing_files(train_json, train_img_dir)
        check_missing_files(val_json, val_img_dir)

        register_coco_instances(train_name, {}, train_json, train_img_dir)
        register_coco_instances(val_name, {}, val_json, val_img_dir)

        fold_mapping[fold_id] = {"train" : train_name, "val": val_name}

        print(f"✅ Registered: {train_name}, {val_name}")

    return fold_mapping

