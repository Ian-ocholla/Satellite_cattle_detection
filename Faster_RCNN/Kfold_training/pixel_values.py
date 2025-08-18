import cv2
import numpy as np
import os

def compute_image_statistics(image_dir):
    means = []
    stds = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = img / 255.0  # Scale pixel values to [0, 1]
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))

    means = np.array(means)
    stds = np.array(stds)
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    print(f"Pixel Mean: {mean * 255}")
    print(f"Pixel Std: {std * 255}")
image_dir ="/projappl/project_2006327/Detectron/2025/satellite_dataset/yolo_data/train_data/images/train"
compute_image_statistics(image_dir)
