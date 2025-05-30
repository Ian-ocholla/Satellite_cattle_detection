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
import evaluation
from evaluation import *

# the sensitivity is only done to assess the best parameters for use, not necessarily a step for every data. To get the best weights and LR only


# Paths to directory
Folder = "/scratch/project_2008354/Cattle_UNet"  # Define the correct folder path
Data_folder = os.path.join(Folder, "SampleData")
weight_path = os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan/pretrained_weights')

# Function to check if a path exists
def check_path(path, path_type="directory"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The {path_type} does not exist: {path}")

# Check if essential paths exist
check_path(Folder, "main folder")
check_path(Data_folder, "data folder")
check_path(weight_path, "weights folder")

# Add the core folder to system path
import sys
sys.path.insert(0, os.path.join(Folder, "modules"))

# Import custom modules
import counting
from counting import *

# Calculate the predictions in float datatype ranging between (0,1)
epsilon = 1e-07

# Load the data from the function load_data
# Set load_train=False to only load test data
test_year = 'Lumo_PNEO3_2022Jan'  # Specify the test year
_, _, Xtest, Ytest, _, Ytest_meta = load_data(Data_folder, test_year, load_train=False)

# Validate test data loading
if Xtest is None or Ytest is None:
    raise ValueError("Test data could not be loaded. Check the load_data function and paths.")

# Initialize variables
Ypredict = [0]
predict_float = [0]
num_folds = 10
weight_set = 0.7
lr_set = 1e-4

# Loop through folds
for i in range(num_folds):
    fold_no = i + 1
    print(f"Processing fold {fold_no}...")
    
    
    # Initialize model
    model = unet(pretrained_weights=None, input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), lr=lr_set)
    
    # Construct weight path
    best_path = os.path.join(
        Folder, 
        f'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/best_weights_fold_{fold_no}.weights.h5'
    )
    
    # Check if weight file exists
    check_path(best_path, "weight file")
    
    # Load weights
    model.load_weights(best_path)
    
    # Predict on test data
    predict = model.predict(Xtest)
    if np.max(predict) > 0.05:
        predict = (predict - np.min(predict)) / (np.max(predict) - np.min(predict))
    
    # Save predictions for this fold
    prediction_path = os.path.join(weight_path, f'predict_fold_{fold_no}.npy')  #make twist here
    np.save(prediction_path, predict, allow_pickle=True, fix_imports=True)
    print(f"Fold {fold_no} predictions saved at {prediction_path}.")
    
    # Accumulate predictions
    Ypredict += predict / num_folds
    predict_float += predict / num_folds

    # Cleanup
    del model
    gc.collect()

# Save the average predictions
average_prediction_path = os.path.join(weight_path, 'predict_average.npy') #make the twist here
np.save(average_prediction_path, predict_float, allow_pickle=True, fix_imports=True)
print(f"Average predictions saved at {average_prediction_path}.")
