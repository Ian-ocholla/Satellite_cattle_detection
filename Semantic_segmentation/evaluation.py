import os
import numpy as np
import gc
import logging
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
import data_processing
from data_processing import load_data
import unet_model
from unet_model import unet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
epsilon = 1e-06
cluster_size = 25 #from 9
num_folds = 10 # match to training and sensitivity files
weight_set = 0.7 #must match with the unet and sensitivity files to load the weights
LEARNING_RATE = 1.0e-04
lr_set = LEARNING_RATE
PATCHSIZE = 336  # Set your patch size
NUMBER_BANDS = 3  # Set number of bands
DROP_OUT = 0

# Paths to directory
Folder = "/scratch/project_2008354/Cattle_UNet"  # Define the correct folder path
Data_folder = os.path.join(Folder, "SampleData")

# Add the core folder to system path
import sys
sys.path.insert(0, os.path.join(Folder, "modules"))

# Import custom modules

import counting
from counting import ImageToPoints, evaluation

weight_path = os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan/pretrained_weights') #from pretrained_weights
point_path = os.path.join(Data_folder, "data_Lumo_PNEO3_2022Jan/3_Train_test/test/10folds/Lumo_PNEO3_test/")


# Ensure the point_path directory exists
os.makedirs(point_path, exist_ok=True)


#set load_train=False to only load test data
test_year = 'Lumo_PNEO3_2022Jan' #specify the test year
_, _, Xtest, Ytest, _, Ytest_meta = load_data(Data_folder, test_year, load_train=False) #change from Folder

# Initialize predictions and placeholders
Ypredict = np.zeros_like(Ytest, dtype=np.float32) #change from Xtest
predict_float = np.zeros_like(Ytest, dtype=np.float32) # from Xtest

# Model testing across folds
for i in range(num_folds):
    
    fold_no = i + 1
    logger.info(f"Processing fold {fold_no}")
    
    #with strategy.scope():
    # Load model
    model = unet(input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), lr=LEARNING_RATE, drop_out=DROP_OUT)  # Ensure drop_out is defined or set
    
    best_path = os.path.join(weight_path, f'best_weights_fold_{fold_no}.weights.h5')
    fallback_path = os.path.join(weight_path, f'weights_fold_{fold_no}.weights.h5')
    
    
    if os.path.exists(best_path):
        
        weight_to_load = best_path
    elif os.path.exists(fallback_path):
        weight_to_load = fallback_path
        logger.warning(f"Best weights for fold {fold_no} not found. Using checkpoint weights at {fallback_path}.")
    else:
        logger.warning(f"No weights found for fold {fold_no}. Skipping this fold.")
        continue  # Skip to the next fold
    
    try:
        model.load_weights(weight_to_load)
        logger.info(f"Loaded weights for fold {fold_no} from {weight_to_load}")
    except Exception as e:
        logger.error(f"Error loading weights for fold {fold_no} from {weight_to_load}: {e}")
        
        
        del model
        gc.collect()
        K.clear_session()
        continue  # Skip to the next fold
    
    # Predict on the test set
    try:
        predict = model.predict(Xtest, batch_size=5)  # Adjust batch_size as needed
        logger.info(f"Predicted on fold {fold_no}")
    except tf.errors.ResourceExhaustedError as e:
        logger.error(f"ResourceExhaustedError during prediction for fold {fold_no}: {e}")
        logger.info("Attempting to predict with a smaller batch size.")
        try:
            predict = model.predict(Xtest, batch_size=2)
            logger.info(f"Predicted on fold {fold_no} with smaller batch size")
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Failed to predict for fold {fold_no} due to memory constraints: {e}")
            del model
            gc.collect()
            K.clear_session()
            continue  # Skip to the next fold
    
    # Normalize predictions
    epsilon_norm = 1e-8
    if np.max(predict) > 0.05:
        predict = (predict - np.min(predict)) / (np.max(predict) - np.min(predict) + epsilon_norm)
        logger.info(f"Normalized predictions for fold {fold_no}")
    
    # Aggregate predictions across folds
    Ypredict += predict / num_folds
    predict_float += predict / num_folds
    
    # Clean up memory
    del model
    gc.collect()
    K.clear_session()

# Replace NaN and infs with zeros to ensure clean data for evaluation
try:
    Ypredict = np.nan_to_num(Ypredict, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info("Replaced NaNs and Infs in Ypredict with zeros")
except Exception as e:
    logger.error(f"Error during np.nan_to_num: {e}")
    Ypredict = np.zeros_like(Ypredict, dtype=np.float32)  # Fallback

# Binarize predictions
Ypredict = (Ypredict > 0.5).astype(np.float32)
logger.info("Binarized predictions based on threshold 0.5")

# Initialize performance metrics
Total_TP = 0
Total_FP = 0
Total_FN = 0

# Loop through each test image and evaluate predictions
for i in range(len(Ytest)):
    logger.info(f"Evaluating image {i+1}/{len(Ytest)}")
    
    # Convert the ground truth and predicted masks into points
    try:
        true_pts = ImageToPoints(Ytest[i], Ytest_meta[i], animal_size=cluster_size)
        predict_pts = ImageToPoints(Ypredict[i], Ytest_meta[i], animal_size=cluster_size)
    except Exception as e:
        logger.error(f"Error converting masks to points for image {i+1}: {e}")
        continue  # Skip to the next image
    
    # Evaluate predictions using your custom evaluation function
    try:
        accuracy = evaluation(
            true_pts, 
            predict_pts, 
            threshold=0.72, # need to change this
            index=str(i+1), 
            ShapefilePath=point_path, 
            meta=Ytest_meta[i]  # Corrected access
        )
        logger.info(f"Image {i+1} - TP: {accuracy['TP']}, FP: {accuracy['FP']}, FN: {accuracy['FN']}")
    except AttributeError as e:
        logger.error(f"AttributeError during evaluation of image {i+1}: {e}")
        accuracy = {"TP": 0, "FP": 0, "FN": 0}  # Fallback
    except Exception as e:
        logger.error(f"Error during evaluation of image {i+1}: {e}")
        accuracy = {"TP": 0, "FP": 0, "FN": 0}  # Fallback
    
    # Accumulate True Positives, False Positives, and False Negatives
    Total_TP += accuracy.get('TP', 0)
    Total_FP += accuracy.get('FP', 0)
    Total_FN += accuracy.get('FN', 0)

# Calculate precision, recall, and F1-score with conditional checks
if (Total_TP + Total_FP) == 0:
    Total_precision = 1.0  # Define as 1 if no positive predictions were made
    logger.warning("Total_TP + Total_FP is zero. Setting Precision to 1.0")
else:
    Total_precision = Total_TP / (Total_TP + Total_FP + epsilon)

if (Total_TP + Total_FN) == 0:
    Total_recall = 1.0  # Define as 1 if there are no actual positives
    logger.warning("Total_TP + Total_FN is zero. Setting Recall to 1.0")
else:
    Total_recall = Total_TP / (Total_TP + Total_FN + epsilon)

if (Total_precision + Total_recall) == 0:
    Total_f1 = 0.0  # Define F1-score as 0 if both precision and recall are 0
    logger.warning("Total_precision + Total_recall is zero. Setting F1-score to 0.0")
else:
    Total_f1 = 2 * (Total_precision * Total_recall) / (Total_precision + Total_recall + epsilon)

# Output performance metrics
logger.info(f"Precision: {Total_precision:.4f}")
logger.info(f"Recall: {Total_recall:.4f}")
logger.info(f"F1-score: {Total_f1:.4f}")

print(f"Precision: {Total_precision:.4f}")
print(f"Recall: {Total_recall:.4f}")
print(f"F1-score: {Total_f1:.4f}")