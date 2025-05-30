import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
import data_processing
from data_processing import load_data
import unet_model
from unet_model import unet
import evaluation
from evaluation import*

# Paths to directory
Folder = "/scratch/project_2008354/Cattle_UNet" 
Data_folder = os.path.join(Folder, "SampleData")

# Add the core folder to system path
import sys
sys.path.insert(0, os.path.join(Folder, "modules"))
import counting
from counting import dataset_evaluation

# Load the data from the function load_data
#set load_train=False to only load test data
test_year = 'Lumo_PNEO3_2022Jan' 
_, _, Xtest, Ytest, _, Ytest_meta = load_data(Data_folder, test_year, load_train=False) 

# Model parameters
num_folds = 10
lr_set = 1e-4
weight_set = 0.7

# Initialize lists for storing results
threshold = []
precision_curve = []
recall_curve = []
f1_curve = []

# Load predictions for each fold
fold_predictions = []
for fold in range(1, 11): #change from num_folds to 1,6
    fold_file = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/predict_fold_{fold}.npy')
    fold_predictions.append(np.load(fold_file))

# Load average predictions
Ypredict_avg = np.load(os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/predict_average.npy'))

# Function to evaluate and save results for a set of predictions
def evaluate_predictions(Ypredict, filename):
    threshold = []
    precision_curve = []
    recall_curve = []
    f1_curve = []

    for t in range(1, 100, 1):
        t = np.float32(t) / 100
        threshold.append(t)

        # Apply threshold to predictions
        binary_predictions = (Ypredict > t).astype(int)

        # Calculate evaluation metrics
        accuracy_curve = dataset_evaluation(binary_predictions, Ytest, Ytest_meta)
        precision_curve.append(accuracy_curve['Precision'])
        recall_curve.append(accuracy_curve['Recall'])
        f1_curve.append(accuracy_curve['F1'])

    # Add boundary conditions for thresholds 0 and 1
    threshold.append(0)
    precision_curve.append(0)
    recall_curve.append(1)
    f1_curve.append(0)

    threshold.append(1)
    precision_curve.append(1)
    recall_curve.append(0)
    f1_curve.append(0)

    # Create and save results DataFrame
    curve = {
        'threshold': threshold,
        'precision': precision_curve,
        'recall': recall_curve,
        'f1': f1_curve
    }
    results = pd.DataFrame(curve)
    results.to_csv(filename, index=False)

# Evaluate and save results for each fold
for fold, fold_prediction in enumerate(fold_predictions):
    fold_csv = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/PR_fold_{fold}.csv')
    evaluate_predictions(fold_prediction, fold_csv)

# Evaluate and save results for average predictions
average_csv = os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/PR_average.csv')
evaluate_predictions(Ypredict_avg, average_csv)
