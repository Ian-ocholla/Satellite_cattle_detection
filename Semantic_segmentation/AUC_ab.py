import os
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

# Paths to directory
Folder = "/scratch/project_2008354/Cattle_UNet" 

# Function to calculate and save AUC for a given fold
def calculate_auc(file_path, fold):
    d = pd.read_csv(file_path)
    d = d.sort_values(by='recall')

    # Calculate the Area Under precision-recall Curve
    auc = np.trapz(y=d['precision'], x=d['recall'])
    d = d.assign(AUC=auc, fold=fold)

    print(f'Fold {fold} AUC=', auc)

    # Save updated data back to CSV
    d.to_csv(file_path, index=False)

# Calculate AUC for each fold
for fold in range(10):
    fold_file = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/PR_fold_{fold}.csv')
    calculate_auc(fold_file, fold)

# Calculate AUC for the average predictions
average_file = os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan/pretrained_weights/PR_average.csv')
calculate_auc(average_file, fold=0)
