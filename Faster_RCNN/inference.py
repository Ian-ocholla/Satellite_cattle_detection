import os
import cv2
import pickle
import torch
import json
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def load_model(cfg_path, weights_path):
    """
    Loads the trained Detectron2 model.
    """
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Set confidence threshold

    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def calculate_iou(pred_box, gt_box):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        pred_box (tuple): The predicted bounding box (x_min, y_min, x_max, y_max).
        gt_box (tuple): The ground truth bounding box (x_min, y_min, x_max, y_max).
    
    Returns:
        float: IoU score between the predicted and ground truth bounding boxes.
    """
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_box

    # Calculate intersection
    x_min_inter = max(x_min_pred, x_min_gt)
    y_min_inter = max(y_min_pred, y_min_gt)
    x_max_inter = min(x_max_pred, x_max_gt)
    y_max_inter = min(y_max_pred, y_max_gt)

    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate areas
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)

    # Calculate union
    union_area = area_pred + area_gt - intersection_area

    # Return IoU
    return intersection_area / union_area if union_area != 0 else 0

def get_ground_truth_boxes(image_path, annotation_file="coco_data/val/_annotations.coco.json"):
    """
    Retrieve the ground truth bounding boxes for a given image from a single coco-style JSON file.
    
    Args:
        image_path (str): Path to the input image.
        annotation_file (str): Path to the coco JSON file containing all annotations for the images.
    
    Returns:
        list: List of ground truth bounding boxes in the form (x_min, y_min, x_max, y_max).
    """
    # Get the image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load the annotation file
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file {annotation_file} not found.")
    
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    
    # Extract image ID from the annotation data
    image_id = None
    for img_info in annotations['images']:
        if img_info['file_name'] == os.path.basename(image_path):
            image_id = img_info['id']
            break

    if image_id is None:
        print(f"Warning: No annotations found for image {image_name}.")
        return []

    # Find ground truth bounding boxes for the corresponding image ID
    ground_truth_boxes = []
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']  # [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            ground_truth_boxes.append((x_min, y_min, x_max, y_max))
    
    return ground_truth_boxes

def run_inference(image_path, predictor, output_dir, conf_threshold=0.4, iou_threshold=0.2):
    """
    Runs inference on a single image, compares predictions with ground truth, 
    and saves YOLO-style bounding boxes to a .txt file.
    
    Args:
        image_path (str): Path to the input image.
        predictor: Detectron2 predictor object.
        output_dir (str): Path to save YOLO label files.
        conf_threshold (float): Confidence threshold for filtering predictions.
        iou_threshold (float): IoU threshold for determining TP/FP.
    
    Returns:
        tuple: TP, FP, FN for the current image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to read {image_path}")

    height, width, _ = image.shape  # Image dimensions

    # Perform inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Extract bounding boxes, scores, and class IDs
    boxes = instances.pred_boxes.tensor.numpy()  # (x_min, y_min, x_max, y_max)
    scores = instances.scores.numpy()  # Confidence scores
    classes = instances.pred_classes.numpy().astype(int)  # Convert to integer class IDs

    # Get ground truth bounding boxes
    ground_truth_boxes = get_ground_truth_boxes(image_path)

    tp, fp, fn = 0, 0, 0
    # For each predicted bounding box, check against ground truth
    for box, score, class_id in zip(boxes, scores, classes):
        if score < conf_threshold:
            continue

        # Check IoU against ground truth boxes
        matched = False
        for gt_box in ground_truth_boxes:
            iou = calculate_iou(box, gt_box)
            if iou >= iou_threshold:
                tp += 1  # True Positive if IoU exceeds threshold
                matched = True
                break
        
        if not matched:
            fp += 1  # False Positive if no match is found with IoU >= threshold
    
    # Count false negatives (FN)
    fn = len(ground_truth_boxes) - tp  # FN is the number of ground truth boxes that weren't matched

    # Save YOLO-format bounding boxes (same as before)
    yolo_bboxes = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score >= conf_threshold:
            x_min, y_min, x_max, y_max = box
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            yolo_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Define output file path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_output_path = os.path.join(output_dir, f"{image_name}.txt")

    # Save YOLO-format bounding boxes
    if yolo_bboxes:
        os.makedirs(output_dir, exist_ok=True)
        with open(txt_output_path, "w") as f:
            f.write("\n".join(yolo_bboxes))
        print(f"Saved: {txt_output_path}")
    else:
        print(f"No valid detections for {image_path}")

    # Print TP/FP/FN count for evaluation
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")

    # Calculate precision and recall for the current image
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Print precision and recall for the current image
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    return tp, fp, fn


def evaluate_all_images(input_dir, predictor, output_dir, conf_threshold=0.4, iou_threshold=0.2):
    """
    Evaluates all images in the input directory and calculates total TP, FP, FN, precision, and recall.

    Args:
        input_dir (str): Directory containing images.
        predictor: Detectron2 predictor object.
        output_dir (str): Directory to save YOLO label files.
        conf_threshold (float): Confidence threshold for filtering predictions.
        iou_threshold (float): IoU threshold for determining TP/FP.
    
    Returns:
        dict: A dictionary with total TP, FP, FN, precision, and recall.
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    # Process all images in the input directory
    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_dir, image_file)
            tp, fp, fn = run_inference(image_path, predictor, output_dir, conf_threshold, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Calculate overall precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # Print total metrics
    print(f"\nTotal TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    print(f"Total Precision: {precision:.4f}, Total Recall: {recall:.4f}")

    return {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'precision': precision,
        'recall': recall
    }

# Set the working directory
WORK_DIR = "/projappl/project_2006327/Detectron/2025/satellite_dataset"  # Change this to your actual project directory
os.chdir(WORK_DIR)  # Change the working directory globally

if __name__ == "__main__":
    cfg_path = "output/A001/coco_data_test/cfg.pickle"
    weights_path = "output/A001/coco_data_test/model_best.pth"
    input_dir = "coco_data/val/"  # Folder containing images
    #input_dir ="/projappl/project_2006327/Detectron/2025/satellite_test_dataset/Choke_PNEO/coco_data/test"
    output_dir = "output/yolo_labels/val/iou02/cf04/labels/"  # Folder for saving YOLO labels

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Load model
    cfg, predictor = load_model(cfg_path, weights_path)

    # Evaluate all images in the input directory
    results = evaluate_all_images(input_dir, predictor, output_dir)

    # Print total precision and recall
    print(f"\nTotal Precision: {results['precision']:.4f}, Total Recall: {results['recall']:.4f}")

    # Process all images in the input directory
    #for image_file in os.listdir(input_dir):
    #    if image_file.lower().endswith((".jpg", ".png", ".jpeg")):
    #        image_path = os.path.join(input_dir, image_file)
    #        run_inference(image_path, predictor, output_dir)

