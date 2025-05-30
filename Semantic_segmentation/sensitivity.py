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

#load the functions from py files
import unet_model 
from unet_model import * 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#use tensflow memory growth feature
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logger.error(e)


# Sensitivity analysis parameters
weight_FP = [0.6, 0.7, 0.8, 0.9]
#weight_FP = [0.7]  # Set weight of FP in Tversky loss from 0.9 to 0.6

#Learning rate and dropout rate
#lr = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
LEARNING_RATE = 1.0e-04  # Learning rate
lr_set = LEARNING_RATE
DROP_SET = 0  # Dropout rate
EPOCH_NUMBERS = 25
#BATCH_SIZE= 12

GLOBAL_BATCH_SIZE = 8

BATCH_SIZE = GLOBAL_BATCH_SIZE// strategy.num_replicas_in_sync

#K-fold cross validation
num_folds = 10  # K-Fold Cross Validation
PATCHSIZE = 336  # Set your patch size
NUMBER_BANDS = 3  # Number of bands in your input data

#Define folders
Folder = "/scratch/project_2008354/Cattle_UNet"

# Dictionary to store results
results_dic = {
    "weight": [],
    "learning_rate": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

epsilon = 1e-06  # A small constant to avoid division by zero

def tversky_loss(alpha, beta, epsilon=1e-7): #was alpha=1-weight_set and beta=weight_set
    """
    Function to calculate the Tversky loss for imbalanced data
    Args:
        y_true: Ground truth segmentation mask
        y_pred: Predicted logits
        alpha: Weight of false negatives
        beta: Weight of false positives
    Returns:
        Tversky loss value
    """
    #cast y_ture and y_pred to float32 to ensure consistent types
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        y_pred = K.cast(y_pred, dtype='float32')
        #y_pred = K.clip(y_pred, epsilon, 1- epsilon) #clip y_pred to [epsilon, 1-epsilon]

        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        #print(f"true_pos: {true_pos}, false_neg: {false_neg}, false_pos: {false_pos}")
    
        return 1 - (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    return loss


#Define a prediction function to minimize tf.function retracing
@tf.function
def predict_step(model, data):
    return model(data, training=False)
    
#iterate over each weight in weight_FP
for weight_i in weight_FP:

    alpha = 1- weight_i
    beta = weight_i
    logger.info(f"Evaluating with weight_FP: {weight_i} (alpha: {alpha}, beta: {beta})")

    #initialize K-fold
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 1 # initialize fold number
    

    for train_idx, val_idx in kfold.split(Xtrain, Ytrain):
        logger.info(f"starting training for fold {fold_no} with weight {weight_i}")
        
        # data augmentation
        xtrain, ytrain = augment_data(Xtrain[train_idx], Ytrain[train_idx])
        xval, yval = np.float32(Xtrain[val_idx]), np.float32(Ytrain[val_idx])

        # Define model checkpoint path
        check_path = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan_GPUs_b8_sens/sensitivity_analysis/weights_lr_{lr_set}_weight_{weight_i}_fold_{fold_no}.weights.h5') #*** _fold{j}
        checkpoint = ModelCheckpoint(check_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min', 
            save_weights_only=True)

        logger.info(f"Starting training for fold {fold_no} with weight {weight_i}")

        # Define learning rate reducer
        reduceLROnPlat = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.33, 
            patience=0, 
            verbose=1, 
            mode='min', 
            min_delta=0.0001, 
            cooldown=4,
            min_lr=1e-16)
        callbacks_list = [checkpoint, reduceLROnPlat]
        
        # Build and compile the U-Net model
        model = unet(input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), lr=LEARNING_RATE, drop_out=DROP_SET)
        model.compile(optimizer='adam', loss=tversky_loss(alpha, beta), metrics=['accuracy'])

        hist = None
        # Train the model
        try:
            hist = model.fit(
                    x=xtrain, 
                    y=ytrain, 
                    epochs=EPOCH_NUMBERS, 
                    batch_size=BATCH_SIZE, 
                    validation_data=(xval, yval), 
                    callbacks=callbacks_list, 
                    verbose=1) #check the epochs and batch sizes
                
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"ResourceExhaustedError during training for fold {fold_no}: {e}")
            logger.info("Consider reducing the batch size or using a smaller model.")
            # optimally, implement retry logic with smaller batch size here
            
            #clean up memeory before continuing
            del model
            del hist
            gc.collect()
            K.clear_session()
            fold_no += 1
            continue #skip to next fold
        
         # Define loss plot path
        loss_plot_path = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan_GPUs_b8_sens/sensitivity_analysis/loss_plot_fold_{fold_no}.png')
            
        if hist is not None:
        
            # Plot and save loss history
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title(f'Model loss (Fold {fold_no})')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(loss_plot_path) 
            plt.close() #close the plot to free memory
            logger.info(f"Saved loss for fold {fold_no} at {loss_plot_path}")
            #plt.clf()  # Clear the plot for the next fold

            # Best loss and epoch
            best_loss = np.min(hist.history['val_loss'])
            best_epoch = hist.history['val_loss'].index(best_loss) + 1
            logger.info(f"Best loss for fold {fold_no}: {best_loss} at epoch {best_epoch}")
        else:
            logger.warning(f"Skipping loss plot for fold {fold_no} due to missing training history.")
        
        # If the model's performance is good, save the best weights
        if best_loss < 0.95: #Adjust this threshold
            best_weights_path = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan_GPUs_b8_sens/sensitivity_analysis/best_weights_lr_{lr_set}_weight_{weight_i}_fold_{fold_no}.weights.h5')
            try:
                os.rename(check_path, best_weights_path)
                logger.info(f"Saved best weights for fold {fold_no} at {best_weights_path}")
            except FileNotFoundError:
                logger.error(f"Checkpoint file not found at {check_path}. Cannot rename to {best_weights_path}.")
        else:
            logger.warning(f"Fold {fold_no}: Loss did not improve significantly (best_loss: {best_loss}). Consider retraining.")
        

        # Clean up memory and increase fold number
        del model
        del hist #new inclusion
        gc.collect()
        K.clear_session()

        fold_no += 1  # Increment fold number

    # After all folds, evaluate across all folds
    Ypredict = np.zeros_like(Ytest, dtype=np.float32) #change from Xtest
    
    # List available weight files for debugging
    sensitivity_dir = os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan_GPUs_b8_sens/sensitivity_analysis')
    weight_files = os.listdir(sensitivity_dir)
    logger.info(f"Available weight files: {weight_files}")
    
    
    for j in range(1, num_folds + 1):
        # model = unet(input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), lr=lr_set, drop_out=drop_set)
        best_weights_path = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan_GPUs_b8_sens/sensitivity_analysis/best_weights_lr_{lr_set}_weight_{weight_i}_fold_{j}.weights.h5')
        
        if not os.path.exists(best_weights_path):
            logger.warning(f"Best weights for fold {j} not found at {best_weights_path}. Skipping this fold.")
            continue # skip to next fold
        
        
        logger.info(f"Loading weights from fold {j} for prediction.")
        
        # Build and compile the U-Net model for prediction
        model = unet(input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), lr=LEARNING_RATE, drop_out=DROP_SET)
        model.compile(optimizer='adam', loss=tversky_loss(alpha, beta), metrics=['accuracy'])
        
        try:
            model.load_weights(best_weights_path)
        except Exception as e:
            logger.error(f"Error loading weights from {best_weights_path}: {e}")
            del model
            gc.collect()
            K.clear_session()
            continue  # Skip to next fold
        
        # Predict with reduced batch size to manage memory
        try:
            predict = model.predict(Xtest, batch_size=5)  # Reduced batch size
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"ResourceExhaustedError during prediction for fold {j}: {e}")
            logger.info("Attempting to predict with an even smaller batch size.")
            try:
                predict = model.predict(Xtest, batch_size=2)
            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"Failed to predict for fold {j} due to memory constraints: {e}")
                del model
                gc.collect()
                K.clear_session()
                continue  # Skip this fold

        # Normalize predictions
        epsilon_norm = 1e-8
        if np.max(predict) > 0.05:
            predict = (predict - np.min(predict)) / (np.max(predict) - np.min(predict) + epsilon_norm)
            
        Ypredict += predict / num_folds # Aggregate predictions
        
        # Clean up memeory
        del model
        gc.collect()
        K.clear_session()

    # Binarize predictions
    ## Ypredict[Ypredict > 0.5] = 1.0
    ## Ypredict[Ypredict <= 0.5] = 0.0
    
    #Replace NaN and infs with zeros to ensure clean adata for evaluation
    Ypredict = np.nan_to_num(Ypredict, nan=0.0, posinf=0.0, neginf=0.0)
    
    #Binarize predictions
    Ypredict = (Ypredict > 0.5).astype(np.float32)

    # Evaluate the predictions
    ## eval_dic = dataset_evaluation(Ypredict, Ytest)
    try:
        eval_dic = dataset_evaluation(Ypredict, Ytest, Ytest_meta)  # Passed Ytest_meta as meta_list
    except TypeError as e:
        logger.error(f"TypeError during dataset evaluation: {e}")
        logger.info("Ensure that dataset_evaluation is defined to accept three arguments: Ypredict, Ytest, and meta_list.")
        continue  # Skip to next weight if evaluation fails
    
    
   
    # Append evaluation results to results_dic
    results_dic["weight"].append(weight_i)
    results_dic["learning_rate"].append(lr_set)
    results_dic["precision"].append(eval_dic.get("precision", None))
    results_dic["recall"].append(eval_dic.get("recall", None))
    results_dic["f1_score"].append(eval_dic.get("f1_score", None))

    # Save evaluation results for the current configuration
    results_df = pd.DataFrame(results_dic)
    results_csv_path = os.path.join(Folder, f'tmp/Lumo_PNEO3_2022Jan_GPUs_b8_sens/sensitivity_analysis/PT_result_weight_{weight_i}_learning_rate_{lr_set}.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved evaluation results at {results_csv_path}")

logger.info("Final Results:")
logger.info(results_dic)