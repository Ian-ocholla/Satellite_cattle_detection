import os
import cv2
import pickle
import torch
import json
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
from ensemble_boxes import weighted_boxes_fusion

"""
Conducts the WBF ensembling using the IOU and confidence thresholds

This is for validation purposes, it requires annotation file

"""

def load_model(cfg_path, weights_path):
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    return DefaultPredictor(cfg)

def calculate_iou(pred_box, gt_box):
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_box
    x_min_inter = max(x_min_pred, x_min_gt)
    y_min_inter = max(y_min_pred, y_min_gt)
    x_max_inter = min(x_max_pred, x_max_gt)
    y_max_inter = min(y_max_pred, y_max_gt)
    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)
    union_area = area_pred + area_gt - intersection_area
    return intersection_area / union_area if union_area != 0 else 0

def get_ground_truth_boxes(image_path, annotation_file):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    image_id = None
    for img_info in annotations['images']:
        if img_info['file_name'] == os.path.basename(image_path):
            image_id = img_info['id']
            break
    if image_id is None:
        return []
    ground_truth_boxes = []
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_id:
            x_min, y_min, width, height = annotation['bbox']
            x_max = x_min + width
            y_max = y_min + height
            ground_truth_boxes.append((x_min, y_min, x_max, y_max))
    return ground_truth_boxes

def ensemble_inference(image_path, predictors, conf_threshold=0.2):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    boxes_list, scores_list, labels_list = [], [], []
    for predictor in predictors:
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        labels = instances.pred_classes.numpy().astype(int)
        norm_boxes = [[x1 / width, y1 / height, x2 / width, y2 / height] for x1, y1, x2, y2 in boxes]
        boxes_list.append(norm_boxes)
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=conf_threshold)
    final_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        final_boxes.append((x1, y1, x2, y2, score, label))
    return final_boxes, image.shape

def run_and_evaluate(image_path, predictors, annotation_file, output_dir, conf_threshold=0.4, iou_threshold=0.2):
    boxes, image_shape = ensemble_inference(image_path, predictors, conf_threshold)
    height, width = image_shape[:2]
    ground_truth_boxes = get_ground_truth_boxes(image_path, annotation_file)
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    for box in boxes:
        x1, y1, x2, y2, score, label = box
        if score < conf_threshold:
            continue
        pred_box = (x1, y1, x2, y2)
        matched = False
        for i, gt_box in enumerate(ground_truth_boxes):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(ground_truth_boxes) - len(matched_gt)
    yolo_bboxes = []
    for box in boxes:
        x1, y1, x2, y2, score, label = box
        if score >= conf_threshold:
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            yolo_bboxes.append(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_output_path = os.path.join(output_dir, f"{image_name}.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(txt_output_path, "w") as f:
        f.write("\n".join(yolo_bboxes))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return tp, fp, fn, precision, recall

def evaluate_folder(image_dir, predictors, annotation_file, output_dir, conf_threshold=0.3, iou_threshold=0.2):
    total_tp, total_fp, total_fn = 0, 0, 0
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(".png"):
            image_path = os.path.join(image_dir, image_file)
            tp, fp, fn, precision, recall = run_and_evaluate(image_path, predictors, annotation_file, output_dir, conf_threshold, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    print(f"Total Precision: {total_precision:.4f}, Total Recall: {total_recall:.4f}")
    return {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'precision': total_precision,
        'recall': total_recall
    }

if __name__ == "__main__":
    WORK_DIR = "/projappl/project_2006327/Detectron/2025/Kfold_dataset/train_aug_data"
    os.chdir(WORK_DIR)
    fold_paths = [f"output/train_01/fold_{i}_val/fold_{i}" for i in range(5)]
    predictors = []
    for fold_path in fold_paths:
        cfg_path = os.path.join(os.path.dirname(fold_path), "cfg.pickle")
        weights_path = os.path.join(fold_path, "model_best.pth")
        predictor = load_model(cfg_path, weights_path)
        predictors.append(predictor)
    input_dir = "/scratch/project_2006327/Detectron_data_2025/satellite_dataset/coco_data/test" #inference dataset dir
    annotation_file = "/scratch/project_2006327/Detectron_data_2025/satellite_dataset/coco_data/test/_annotations.coco.json"
    output_dir = "output/yolo_labels_01/test/LM_PNEO/2/cf3"
    evaluate_folder(input_dir, predictors, annotation_file, output_dir)

