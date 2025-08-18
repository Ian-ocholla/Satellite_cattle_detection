#convert COCO JSON into k folds

from sklearn.model_selection import KFold
import json
import os
import shutil

# Set the working directory
WORK_DIR = "/projappl/project_2006327/Detectron/2025/Kfold_dataset/train_aug_data"  # Change this to your actual project directory
os.chdir(WORK_DIR)  # Change the working directory globally

# Config
data_path = os.path.join(WORK_DIR, "coco_data/train")  # Contains images/ and annotation.json
output_base = os.path.join(WORK_DIR, "coco_data/Kfolds")
os.makedirs(output_base, exist_ok=True)
num_folds = 5

# Load COCO annotations
with open(os.path.join(data_path, "_annotations.coco.json")) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# check for any missing files
# ✅ Optional: Validate image file paths before continuing
missing_files = []
for img in images:
    path = os.path.join(data_path, img["file_name"])
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    print("⚠️ WARNING: Missing image files:")
    for p in missing_files:
        print(p)
    print(f"❌ Total missing: {len(missing_files)}")
    import sys
    sys.exit(1)
else:
    print("All image files found.")
    # Optional: stop script if many missing files
    # import sys; sys.exit(1)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    fold_dir = os.path.join(output_base, f"fold_{fold}")
    os.makedirs(os.path.join(fold_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "images", "val"), exist_ok=True)

    # Split images
    train_imgs = [images[i] for i in train_idx]
    val_imgs = [images[i] for i in val_idx]

    train_img_ids = {img["id"] for img in train_imgs}
    val_img_ids = {img["id"] for img in val_imgs}

    # Split annotations
    train_annots = [ann for ann in annotations if ann["image_id"] in train_img_ids]
    val_annots = [ann for ann in annotations if ann["image_id"] in val_img_ids]

    # Create train/val annotation.json
    for split, split_imgs, split_anns in [("train", train_imgs, train_annots), ("val", val_imgs, val_annots)]:
        json_data = {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": coco["categories"]
        }
        with open(os.path.join(fold_dir, f"{split}_annotation.json"), "w") as f:
            json.dump(json_data, f)

        # Optionally copy images
        for img in split_imgs:
            src = os.path.join(data_path, img["file_name"])
            dst = os.path.join(fold_dir, "images", split, img["file_name"])

            if not os.path.exists(src):
                print(f" Skipping missing file: {src}")
                continue
                
            shutil.copyfile(src, dst)
