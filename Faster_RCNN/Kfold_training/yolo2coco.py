import os
import shutil
from glob import glob
from tqdm import tqdm
from pylabel import importer
import json

def process_dataset(annotations_path, images_path, coco_dir, classes_file="yolo_data/train_data/classes.txt"):
    """
    Process a dataset (train/test/val) by copying images and annotations, converting to COCO format, 
    fixing duplicate annotation IDs, and removing original annotations.
    """
    os.makedirs(coco_dir, exist_ok=True)
    
    # Get class names
    with open(classes_file, "r") as f:
        classes = f.read().split("\n")
    
    # Get txt and image files
    txt_files = glob(os.path.join(annotations_path, "*.txt"))
    img_files = glob(os.path.join(images_path, "*.jpg")) + glob(os.path.join(images_path, "*.png"))

    # Copy annotations and images to the COCO directory
    for f in tqdm(txt_files, desc="Copying annotations"):
        shutil.copy(f, coco_dir)
    for f in tqdm(img_files, desc="Copying images"):
        shutil.copy(f, coco_dir)

    # Load dataset
    dataset = importer.ImportYoloV5(path=coco_dir, cat_names=classes, name="C")

    # Export to COCO format
    coco_file = os.path.join(coco_dir, "_annotations.coco.json")
    dataset.export.ExportToCoco(coco_file, cat_id_index=1)

    # Load exported COCO JSON
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    # Fix duplicate annotation IDs
    unique_id = 1
    existing_ids = set()
    for ann in coco_data["annotations"]:
        if ann["id"] in existing_ids:
            ann["id"] = unique_id  # Assign a new unique ID
        existing_ids.add(ann["id"])
        unique_id += 1

    # Identify and include negative samples (images without annotations)
    annotated_images = {img["file_name"] for img in coco_data["images"]}  # Track existing images
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        if img_name not in annotated_images:
            coco_data["images"].append({
                "id": len(coco_data["images"]) + 1,
                "file_name": img_name,
                "width": 336,  # Adjust based on actual image sizes
                "height": 336
            })  # No annotations added

    # Save the cleaned JSON
    with open(coco_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Duplicate annotation IDs fixed and COCO dataset updated for {coco_dir}.")

    # Remove original YOLO annotations
    for f in txt_files:
        os.remove(f.replace(annotations_path, coco_dir))

if __name__ == "__main__":
    # Process the train, test, and val datasets
    process_dataset("yolo_data/train_data/labels/train", "yolo_data/train_data/images/train", "coco_data/train")
    process_dataset("yolo_data/train_data/labels/test", "yolo_data/train_data/images/test", "coco_data/test")
    process_dataset("yolo_data/train_data/labels/val", "yolo_data/train_data/images/val", "coco_data/val")
