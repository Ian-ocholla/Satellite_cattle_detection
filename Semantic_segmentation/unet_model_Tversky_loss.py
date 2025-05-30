import sys
import os

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time


# Adjust the folder path
import data_processing
from data_processing import * 

# Define Tversky loss function
epsilon = 1.0e-06

# Define constanats
PATCHSIZE= 336
READ_CHANNEL = [1,2,3] #USE rgb
NUMBER_BANDS = len(READ_CHANNEL)

# Define training parameters
NUMBER_EPOCHS = 50  # Modify as needed
#BATCH_SIZE = 12      # Modify as needed
DROP_OUT = 0
LEARNING_RATE = 1.0e-04
THRESHOLD_LOSS = 0.5
num_folds = 10

# Ensure proper GPU utilization
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPUs initialized:", gpus)


# Enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE = 8

BATCH_SIZE = GLOBAL_BATCH_SIZE// strategy.num_replicas_in_sync

weight_set = 0.7 #least should be 0.7 to ensure beta is greater than alpha

def tversky(y_true, y_pred, alpha=1-weight_set, beta=weight_set):
    
    epsilon = K.epsilon()
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    
    # TP
    true_pos = K.sum(y_true_pos * y_pred_pos)
    # FN
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    # FP
    false_pos = K.sum((1-y_true_pos) * y_pred_pos)
    #print(f"true_pos: {true_pos}, false_neg: {false_neg}, false_pos: {false_pos}")
    
    return 1 - ((true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon))

# Metrics
def accuracy(y_true, y_pred, threshold=THRESHOLD_LOSS):
    
    """compute accuracy"""
    
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    #y_pred = K.round(y_pred +0.5 -threshold)
    #return K.equal(K.round(y_t), K.round(y_pred))
    y_pred = K.cast(y_pred > threshold, K.floatx())
    y_true = K.cast(y_true, K.floatx())
    return K.mean(K.equal(y_true, y_pred), axis=[1, 2, 3])
    
    
def true_positives(y_true, y_pred, threshold=THRESHOLD_LOSS):
    """compute true positives"""
    y_pred = K.round(y_pred +0.5 -threshold) #0.5
    return K.round(y_true * y_pred)
    
def false_positives(y_true, y_pred, threshold=THRESHOLD_LOSS):
    """compute false positive"""
    y_pred = K.round(y_pred +0.5 -threshold) #0.5
    return K.round((1 - y_true) * y_pred)
    
def true_negatives(y_true, y_pred, threshold=THRESHOLD_LOSS):
    """compute true negative"""
    y_pred = K.round(y_pred +0.5 -threshold) #0.5
    return K.round((1 - y_true) * (1- y_pred))
    
def false_negatives(y_true, y_pred, threshold=THRESHOLD_LOSS):
    """compute false negative"""
    y_pred = K.round(y_pred +0.5 -threshold) #0.5
    return K.round((y_true) * (1 - y_pred))
    
def recall_m(y_true, y_pred, threshold=THRESHOLD_LOSS):

    #tp = true_positives(y_true, y_pred)
    #fn = false_negatives(y_true, y_pred)
    #recall = K.sum(tp) / (K.sum(tp) + K.sum(fn) + epsilon)
    #return recall

    y_pred = K.cast(y_pred > threshold, K.floatx())
    y_true = K.cast(y_true, K.floatx())
    tp = K.sum(y_true * y_pred, axis=[1, 2, 3])
    fn = K.sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    return K.mean(tp / (tp + fn + epsilon))
    
def precision_m(y_true, y_pred, threshold=THRESHOLD_LOSS):

    #tp = true_positives(y_true, y_pred)
    #fp = false_positives(y_true, y_pred)
    #precision = K.sum(tp) / (K.sum(tp) + K.sum(fp) + epsilon)
    #return precision

    y_pred = K.cast(y_pred > threshold, K.floatx())
    y_true = K.cast(y_true, K.floatx())
    tp = K.sum(y_true * y_pred, axis=[1, 2, 3])
    fp = K.sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    return K.mean(tp / (tp + fp + epsilon))

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2 * ((precision * recall) / (precision + recall + epsilon))
    

def print_pixel_accuracy(predictions, Ytest):
    print("Pixel-level accuracy: ")
    print("precision_m: ",K.sum(precision_m(Ytest, predictions)))
    print("recall_m: ",K.sum(recall_m(Ytest, predictions)))
    print("f1_m: ",K.sum(f1_m(Ytest, predictions)))

# Define U-Net Model
def unet(pretrained_weights=None, input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), 
         lr=LEARNING_RATE, drop_out=DROP_OUT, regularizers = regularizers.l2(0.0001)): 
    inputs = Input(input_size)
    """
    To avoid information loss. In the encoder- applied two conv block of 3x3 followed by batchnorm and a 2x2 max-pooling layer
    Is there need for ReLu activation
    """
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    norm1 = BatchNormalization()(conv1) #used to normalize activations
    pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(norm2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(norm3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    norm4 = BatchNormalization()(conv4)
    drop4 = Dropout(drop_out)(norm4) #used to prevent overfitting, especially in deeper layers
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(drop_out)(conv5)
    
    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    norm6 = BatchNormalization()(up6)
    merge6 = Concatenate(axis=3)([norm4, norm6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    norm7 = BatchNormalization()(up7)
    merge7 = Concatenate(axis=3)([norm3, norm7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    norm8 = BatchNormalization()(up8)
    merge8 = Concatenate(axis=3)([norm2, norm8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    norm9 = BatchNormalization()(up9)
    merge9 = Concatenate(axis=3)([norm1, norm9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    
    # Output Layer
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    # Compile the Model
    OPTIMIZER = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)
    with strategy.scope():
        model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[accuracy, precision_m, recall_m, f1_m]) #tversky

    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model

# K-fold Cross Validation Training


kfold = KFold(n_splits=num_folds, shuffle=True, random_state=4)

# Initialize lists to store metrics
folds = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
loss_per_fold = []

Val_precision_per_fold = []
Val_recall_per_fold = []
Val_f1_per_fold = []

Test_precision_per_fold = []
Test_recall_per_fold = []
Test_f1_per_fold = []

#split the dataset inot k folds, save the index of training and validation data
split = []
for train, val in kfold.split(Xtrain, Ytrain):
    com = {'train': train, 'val':val}
    split.append(com)


# set the main folder path
Folder = "/scratch/project_2008354/Cattle_UNet"  
weight_path = os.path.join(Folder, "tmp/Lumo_PNEO3_2022Jan_b82/checkpoint")
os.makedirs(weight_path, exist_ok=True) #Ensure the checkpoint dir exists


# train  and evaluation loop

num_folds = len(split)

for fold_no in range(1, num_folds + 1):
    i = fold_no - 1
    
    train = split[i]['train']
    val = split[i]['val']
    
    #augment the dataset
    xtrain, ytrain = augment_data(Xtrain[train], Ytrain[train])
    xval = np.float32(Xtrain[val])
    yval = np.float32(Ytrain[val])

    print(f"Training set shape: {xtrain.shape},{ytrain.shape}")
    print(f"Validation set shape: {xval.shape}, {yval.shape}")
    
    # Create log directory for the current fold
    log_dir = os.path.join(Folder, f"tmp/Lumo_PNEO3_2022Jan_b82/logs/UNet/fold_{fold_no}_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Build and sumarize model 
    with strategy.scope():
        model = unet(pretrained_weights=None, 
                     input_size=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), 
                     lr=LEARNING_RATE, 
                     drop_out=DROP_OUT)
    
    # Define callbacks
    
    callbacks_list = [
        ModelCheckpoint(
            filepath=os.path.join(Folder, 'tmp/Lumo_PNEO3_2022Jan_b82/checkpoint', f"fold_{fold_no}_weights.{{epoch:02d}}-{{val_loss:.4f}}.weights.h5"),
            monitor="val_loss", #late inclusion
            save_weights_only=True, 
            save_best_only=True, #set to true if you prefer saving only the best model per fold
            verbose=1), # to confirm when and where weights are being saved
        ReduceLROnPlateau(monitor='val_loss', 
            factor=0.33, 
            patience=10, 
            verbose=1, 
            mode='min', 
            min_delta=0.0001, 
            cooldown=4, 
            min_lr=1e-16), #change from -16
        TensorBoard(log_dir=log_dir, 
            histogram_freq=0, 
            write_graph=True),
        EarlyStopping(
            monitor='val_loss',
            patience=10, #Number of epochs with no improvement to stop training
            verbose=1,
            mode='min',
            restore_best_weights=True)]
    
    
    
    # print fold info
    print('--------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    #model.summary()
    
    """ changing history to hist"""
    #Train model
    hist = model.fit(x=xtrain,
                     y=ytrain, 
                     batch_size=BATCH_SIZE, #increase to a larger batch size from 5
                     epochs=NUMBER_EPOCHS, 
                     validation_data=(xval, yval), 
                     callbacks=callbacks_list,
                     verbose=1)
    
    #plot the losses
    plt.figure(figsize=(8, 6))
    plt.plot(hist.history['loss'], label ='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold_no} - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(weight_path, f'loss_fold{fold_no}.png'))
    plt.close()

    #plot the validation precision and recall metrics 
    plt.figure(figsize=(8, 6))
    plt.plot(hist.history['val_precision_m'], label ='Validation precision')
    plt.plot(hist.history['val_recall_m'], label='Validation Recall')
    plt.title(f'Fold {fold_no} : Validation P and R')
    plt.ylabel('Metrics')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(weight_path, f'Val_Prec_Rec_fold{fold_no}.png'))
    plt.close()

    # print final training results
    print(f"Fold{fold_no} - Final Training Loss: {hist.history['loss'][-1]}")
    print(f"Fold{fold_no} - Final Validation Loss: {hist.history['val_loss'][-1]}")

    print(f"Fold{fold_no} - Final Validation Precision metric: {hist.history['val_precision_m'][-1]}")
    print(f"Fold{fold_no} - Final Validation Recall metric: {hist.history['val_recall_m'][-1]}")
    print(f"Fold{fold_no} - Final Validation F1-score metric: {hist.history['val_f1_m'][-1]}")
    
    # List saved files in checkpoint directory
    saved_files = os.listdir(weight_path)
    print(f"Files in checkpoint directory after fold {fold_no} training: {saved_files}")

    # Find the best weights based on validation loss
    best_loss = np.min(hist.history['val_loss'])
    best_epoch = hist.history['val_loss'].index(best_loss) + 1  # +1 for epoch numbering
    best_path = os.path.join(weight_path, f"fold_{fold_no}_weights.{best_epoch:02d}-{best_loss:.4f}.weights.h5")

    if best_loss < 0.5:  # Adjust threshold as needed 
        
        # Save the training history
        history_df = pd.DataFrame(hist.history)
        history_df.to_csv(os.path.join(weight_path, f'hist_fold{fold_no}.csv'), index=False)
        
        #best_path = os.path.join(weight_path, f'weights.{best_epoch:02d}-{best_loss:.4f}.weights.h5')
        #print(f'Best weights path: {best_path}')
        
        if os.path.exists(best_path):
            print(f'Best weights path: {best_path}')
            
            #load and rename the best weights
            model.load_weights(best_path)
            new_best_path = os.path.join(weight_path, f'best_weights_fold_{fold_no}.weights.h5')
            os.rename(best_path, new_best_path)
            print(f'Renamed best weights to: {new_best_path}')

            # Evaluate on validation dataset
            scores = model.evaluate(xval, yval, verbose=0)
            print(f'Metrics Names: {model.metrics_names}')
            print(f'Scores: {scores}')

            # Check if required metrics exist
            if len(model.metrics_names) >= 5 and len(scores) >= 5:
                print(f'Score for fold {fold_no}: {model.metrics_names[2]} of {scores[2]:.4f};'
                      f'{model.metrics_names[3]} of {scores[3]:.4f};'
                      f'{model.metrics_names[4]} of {scores[4]:.4f};')

                # Store metrics
                folds.append(fold_no)
                precision_per_fold.append(scores[2] * 100)
                recall_per_fold.append(scores[3] * 100)
                f1_per_fold.append(scores[4] * 100)
                loss_per_fold.append(scores[0])
            else:
                print("Error: Not enough metrics or scores available.")
            
        
            # Validation metrics
            predict_val = model.predict(xval)
            if np.max(predict_val) > 0.05:
                Ypredict_val = (predict_val - np.min(predict_val)) / (np.max(predict_val) - np.min(predict_val))
            else:
                Ypredict_val = np.zeros_like(predict_val)

            Ypredict_val = (Ypredict_val > 0.5).astype(np.float32)
            try:
                dic_val = dataset_evaluation(Ypredict_val, yval, Ytrain_meta[val]) #remove val_index, replace with val
                Val_precision_per_fold.append(dic_val['Precision'])
                Val_recall_per_fold.append(dic_val['Recall'])
                Val_f1_per_fold.append(dic_val['F1'])
            except Exception as e:
                print(f"Error during validation evaluation: {e}")
                Val_precision_per_fold.append(None)
                Val_recall_per_fold.append(None)
                Val_f1_per_fold.append(None)
        
        
            # Testing metrics (assuming Xtest, Ytest, Ytest_meta are defined)
    
            try:
                predict_test = model.predict(Xtest)
                if np.max(predict_test) > 0.05:
                    Ypredict_test = (predict_test - np.min(predict_test)) / (np.max(predict_test) - np.min(predict_test))
                else:
                    Ypredict_test = np.zeros_like(predict_test)
                    
                Ypredict_test = (Ypredict_test > 0.5).astype(np.float32)
                dic_test = dataset_evaluation(Ypredict_test, Ytest, Ytest_meta)
                Test_precision_per_fold.append(dic_test['Precision'])
                Test_recall_per_fold.append(dic_test['Recall'])
                Test_f1_per_fold.append(dic_test['F1'])
            except Exception as e:
                print(f"Error during test evaluation: {e}")
                Test_precision_per_fold.append(None)
                Test_recall_per_fold.append(None)
                Test_f1_per_fold.append(None)
            
        else:
            print(f"Best weights file {best_path} not found. Skipping evaluation for this fold.")
    else:
        print("The loss did not decrease significantly. Retrain this model...")


    
    # clean up to free memory
    del model, hist, xtrain, ytrain, xval, yval
    gc.collect()
    K.clear_session()
    
# After all folds, you can summarize the metrics

for i, fold in enumerate(folds):
    print(f"> Fold {fold}: Loss: {loss_per_fold[i]:.4f}, "
          f"Precision: {precision_per_fold[i]:.2f}, Recall: {recall_per_fold[i]:.2f}, F1: {f1_per_fold[i]:.2f}")
    print(f"  Validation - Precision: {Val_precision_per_fold[i]}, Recall: {Val_recall_per_fold[i]}, F1: {Val_f1_per_fold[i]}")
    print(f"  Testing - Precision: {Test_precision_per_fold[i]}, Recall: {Test_recall_per_fold[i]}, F1: {Test_f1_per_fold[i]}")

metrics_df = pd.DataFrame({
    'Fold': folds,
    'Loss': loss_per_fold,
    'Precision': precision_per_fold,
    'Recall': recall_per_fold,
    'F1': f1_per_fold,
})

metrics_df.to_csv(os.path.join(weight_path, 'fold_metrics_summary.csv'), index=False)
print("Saved fold metrics summary.")  