import os
import glob
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

# === CONFIGURATION ===
PROJECT_DIR = "/scratch/project_2006327/YOLOv5_b"
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets/Lumo_PNEO_data/Kfold_training/kfolds_aug")
PREDICTIONS_ROOT = os.path.join(DATASET_DIR, "runs/detect_MTL/Lumo_PNEO_data/pred_folds_iou05")
TEST_DIR=os.path.join(PROJECT_DIR, "datasets/Lumo_PNEO_data/Test_data_render")
IMAGE_DIR=os.path.join(TEST_DIR,"Lumo_PNEO_data/images/test")
## IMAGE_DIR = os.path.join(DATASET_DIR, "images/test")
OUTPUT_DIR = os.path.join(DATASET_DIR, "runs/detect_MTL/Lumo_PNEO_data/iou05/03/labels")
IMAGE_EXT = ".png"  # or ".jpg"

#PREDICTIONS_ROOT = "/scratch/project_2006327/YOLOv5_b/yolov5/runs/predictions_folds"
#IMAGE_DIR = "/scratch/project_2006327/YOLOv5_b/datasets/Lumo_PNEO_data/Kfold_training/images/test"
#OUTPUT_DIR = "/scratch/project_2006327/YOLOv5_b/datasets/Lumo_PNEO_data/Kfold_training/runs/wbf_output"

"""
IOU_threshold: determines which boxes are grouped togther as prediction of the same object.
It controls how close two boxes must be to be consideered the same detection. Higher values (0.7)
mean more strict grouping-only highly overlapping boxes are merged. Lower values (0.4) mean 
more lenient grouping-boxes with looser overlap are merged. 

skip_box_threshold:  filters out low-confidence prediction before fusion. Any input box with 
a confidence score belw this threshold is ignored completely. This helps remove noise and false positives from affecting the final result
"""

IOU_THRESHOLD = 0.5
SKIP_BOX_THRESHOLD = 0.3
CONF_TYPE = 'max'

# === Create output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Find fold subdirectories and their label dirs ===
fold_dirs = sorted([
    os.path.join(PREDICTIONS_ROOT, d, "labels")
    for d in os.listdir(PREDICTIONS_ROOT)
    if os.path.isdir(os.path.join(PREDICTIONS_ROOT, d, "labels"))
])

print(f"📂 Found folds: {[os.path.basename(os.path.dirname(f)) for f in fold_dirs]}")

# === Get list of image IDs from first fold ===
first_fold = fold_dirs[0]
image_ids = sorted(os.path.basename(f) for f in glob.glob(os.path.join(first_fold, "*.txt")))

# === Helper functions ===
def load_yolo_predictions(txt_path):
    boxes, scores, labels = [], [], []
    if not os.path.exists(txt_path):
        return boxes, scores, labels
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, xc, yc, w, h, conf = map(float, parts)
            x1 = max(xc - w / 2, 0)
            y1 = max(yc - h / 2, 0)
            x2 = min(xc + w / 2, 1)
            y2 = min(yc + h / 2, 1)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(int(cls))
    return boxes, scores, labels

def save_yolo_format(output_path, boxes, scores, labels):
    with open(output_path, 'w') as f:
        for box, score, label in zip(boxes, scores, labels):
            xc = (box[0] + box[2]) / 2
            yc = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            f.write(f"{int(label)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {score:.6f}\n") #included the {int(label)} it was just {label}

# === Main WBF loop ===
for image_id in image_ids:
    boxes_list, scores_list, labels_list = [], [], []

    img_name = image_id.replace(".txt", IMAGE_EXT)
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {img_path}")
        continue

    with Image.open(img_path) as im:
        w, h = im.size

    # Load predictions from each fold
    for labels_dir in fold_dirs:
        txt_path = os.path.join(labels_dir, image_id)
        boxes, scores, labels = load_yolo_predictions(txt_path)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Apply WBF
    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=IOU_THRESHOLD,
        skip_box_thr=SKIP_BOX_THRESHOLD,
        conf_type=CONF_TYPE
    )

    # Save ensembled predictions
    output_path = os.path.join(OUTPUT_DIR, image_id)
    save_yolo_format(output_path, boxes_wbf, scores_wbf, labels_wbf)

#print("✅ WBF ensembling completed.")_path, boxes_wbf, scores_wbf, labels_wbf)

print("✅ WBF ensembling completed for all images.")
