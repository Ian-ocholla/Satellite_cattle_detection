import os
import cv2
import pickle
import numpy as np
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion

# === SETTINGS ===
IOU_THRESHOLD = 0.5          # IoU threshold for WBF
SKIP_BOX_THRESHOLD = 0.5     # Confidence threshold for skipping boxes
IMAGE_EXT = ".png"           # Image file extension

# === MODEL LOADING ===
def load_model(cfg_path, weights_path):
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    return DefaultPredictor(cfg)

# === ENSEMBLE INFERENCE ===
def ensemble_inference(image_path, predictors):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    boxes_list, scores_list, labels_list = [], [], []

    for predictor in predictors:
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        labels = instances.pred_classes.numpy().astype(int)
        norm_boxes = [[x1 / width, y1 / height, x2 / width, y2 / height]
                      for x1, y1, x2, y2 in boxes]
        boxes_list.append(norm_boxes)
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())

    # === Weighted Box Fusion ===
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=IOU_THRESHOLD,
        skip_box_thr=SKIP_BOX_THRESHOLD
    )

    final_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        final_boxes.append((x1, y1, x2, y2, score, label))

    return final_boxes, image.shape

# === SAVE TO YOLO FORMAT ===
def run_inference_only(image_path, predictors, output_dir):
    boxes, image_shape = ensemble_inference(image_path, predictors)
    height, width = image_shape[:2]

    yolo_bboxes = []
    for x1, y1, x2, y2, score, label in boxes:
        if score >= SKIP_BOX_THRESHOLD:
            class_id = int(label)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            yolo_bboxes.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            )

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_output_path = os.path.join(output_dir, f"{image_name}.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(txt_output_path, "w") as f:
        f.write("\n".join(yolo_bboxes))

# === INFERENCE ON FOLDER ===
def inference_folder(image_dir, predictors, output_dir):
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(IMAGE_EXT):
            image_path = os.path.join(image_dir, image_file)
            run_inference_only(image_path, predictors, output_dir)

# === MAIN ===
if __name__ == "__main__":
    WORK_DIR = "/projappl/project_2006327/Detectron/2025/Kfold_dataset/train_aug_data"
    os.chdir(WORK_DIR)

    fold_paths = [f"output/train/fold_{i}_val/fold_{i}" for i in range(10)]
    predictors = []
    for fold_path in fold_paths:
        cfg_path = os.path.join(os.path.dirname(fold_path), "cfg.pickle")
        weights_path = os.path.join(fold_path, "model_best.pth")
        predictors.append(load_model(cfg_path, weights_path))

    input_dir = "/scratch/project_2006327/Detectron_data_2025/satellite_test_data/Lumo_WV3/coco_data/test"
    output_dir = "output/yolo_labels/LM_WV3/iou5_b/05/labels"

    inference_folder(input_dir, predictors, output_dir)

