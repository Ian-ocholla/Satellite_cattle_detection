import sys
import os

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

# Adjust the folder path
import data_processing
from data_processing import * 


# Define Tversky loss function
epsilon = 1e-06

# Define constanats
PATCHSIZE= 336
READ_CHANNEL = [1,2,3] #USE rgb
NUMBER_BANDS = len(READ_CHANNEL)

# Define training parameters
NUMBER_EPOCHS = 50   # Modify as needed
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
GLOBAL_BATCH_SIZE = 32

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
def accuracy(y_true, y_pred, threshold=0.5):
    
    """compute accuracy"""
    
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 -threshold)
    return K.equal(K.round(y_t), K.round(y_pred))
    
def true_positives(y_true, y_pred, threshold=0.5):
    """compute true positives"""
    y_pred = K.round(y_pred +0.5 -threshold)
    return K.round(y_true * y_pred)
    
def false_positives(y_true, y_pred, threshold=0.5):
    """compute false positive"""
    y_pred = K.round(y_pred +0.5 -threshold)
    return K.round((1 - y_true) * y_pred)
    
def true_negatives(y_true, y_pred, threshold=0.5):
    """compute true negative"""
    y_pred = K.round(y_pred +0.5 -threshold)
    return K.round((1 - y_true) * (1- y_pred))
    
def false_negatives(y_true, y_pred, threshold=0.5):
    """compute false negative"""
    y_pred = K.round(y_pred +0.5 -threshold)
    return K.round((y_true) * (1 - y_pred))
    
def recall_m(y_true, y_pred):

    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    recall = K.sum(tp) / (K.sum(tp) + K.sum(fn) + epsilon)
    return recall
    
def precision_m(y_true, y_pred):

    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    precision = K.sum(tp) / (K.sum(tp) + K.sum(fp) + epsilon)
    return precision

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

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

#Define an encoder

def get_encoder(input_tensor):
    # Load ResNet50 with pre-trained ImageNet weights, excluding the top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    #for layer in base_model.layers:
    #    print(layer.name)
    
    # Extract specific layers for skip connections
    # You can choose layers based on the architecture for optimal feature fusion
    layer_names = [
        'conv1_relu',        # 168x168
        'conv2_block3_out',  # 84x84
        'conv3_block4_out',  # 42x42
        'conv4_block6_out',  # 21x21
        'conv5_block3_out'   # 11x11
    ]
    
    layers_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the encoder model
    encoder = models.Model(inputs=base_model.input, outputs=layers_outputs, name='ResNet50_Encoder')
    
    return encoder
    
    
# Define a decoder

def build_decoder(skip_connections, num_classes):
    # Unpack skip connections
    s1, s2, s3, s4, s5 = skip_connections
    
    # Decoder Block 1
    d1 = layers.Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(s5)  # 11x11 -> 22x22
    s4_padded = layers.ZeroPadding2D(((0,1),(0,1)))(s4)                         # 21x21 -> 22x22
    d1 = layers.Concatenate()([d1, s4_padded])                                         # 22x22
    d1 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(d1)
    d1 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(d1)
    d1 = layers.BatchNormalization()(d1) #BatchNorm
    
    # Decoder Block 2
    d2 = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(d1)  # 21x21 -> 42x42
    d2 = layers.Cropping2D(((1,1),(1,1)))(d2)  # 44x44 -> 42x42
    d2 = layers.Concatenate()([d2, s3])  # 42x42
    d2 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(d2)
    d2 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(d2)
    d2 = layers.BatchNormalization()(d2)
    
    # Decoder Block 3
    d3 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(d2)  # 42x42 -> 84x84
    d3 = layers.Concatenate()([d3, s2])  # 84x84
    d3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(d3)
    d3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(d3)
    d3 = layers.BatchNormalization()(d3)
    
    # Decoder Block 4
    d4 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(d3)  # 84x84 -> 168x168
    # No cropping needed if dimensions align
    d4 = layers.Concatenate()([d4, s1])  # 168x168
    d4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(d4)
    d4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(d4)
    d4 = layers.BatchNormalization()(d4)
    
    # Final Upsampling to reach 336x336
    d5 = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(d4)  # 168x168 -> 336x336
    d5 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(d5)
    d5 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(d5)
    
    # Output Layer
    output = layers.Conv2D(num_classes, (1,1), activation='sigmoid')(d5)
    
    return output


# Assemble the U-Net Model

def unet(input_shape=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), lr=LEARNING_RATE, drop_out=DROP_OUT):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    encoder = get_encoder(inputs)
    skip_connections = encoder.output  # List of feature maps from encoder
    
    # Decoder
    decoder_output = build_decoder(skip_connections, num_classes=1)
    
    # Define the model
    model = models.Model(inputs=inputs, outputs=decoder_output, name='U-Net_with_ResNet50_Encoder')
    
    #compile the model
    #alpha = 1 - weight_set
    #beta = weight_set
    
    #Adam optimzer
    optimizer = Adam(learning_rate=1.0e-4, beta_1=0.9, beta_2=0.999, epsilon=1.0e-6)
    with strategy.scope():
        model.compile(optimizer=optimizer,
                      loss=tversky, #(alpha, beta),
                      metrics=[accuracy, precision_m, recall_m, f1_m])
        
    # set SGD optimizer ---results to very high validation loss not working as the better option
    #optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5)
    #sdg_optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True) #use a higher le 1.0e-3 or 1.0e-4
    #compile the model with SGD optimizer
    #with strategy.scope():
    #    model.compile(optimizer=sdg_optimizer,
    #                  loss=tversky, #(alpha, beta),
    #                  metrics=[accuracy, precision_m, recall_m, f1_m])
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

Folder = "/scratch/project_2008354/Cattle_UNet"  # Set the correct folder path for saving weights and logs

weight_path = os.path.join(Folder, "tmp/Lumo_PNEO_ResNet/checkpoint")
os.makedirs(weight_path, exist_ok=True)


# train and evaluation loop

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
    log_dir = os.path.join(Folder, f"tmp/Lumo_PNEO_ResNet/logs/UNet/fold_{fold_no}_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Define callbacks
    
    callbacks_list = [
        ModelCheckpoint(
            filepath=os.path.join(Folder, 'tmp/Lumo_PNEO_ResNet/checkpoint', f"fold_{fold_no}_weights.{{epoch:02d}}-{{val_loss:.4f}}.weights.h5"),
            #monitor="val_loss", #late inclusion
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
            min_lr=1e-6), #change from -16
        TensorBoard(log_dir=log_dir, 
            histogram_freq=0, 
            write_graph=True),
        EarlyStopping(
            monitor='val_loss',
            patience=10, #Number of epochs with no improvement to stop training
            verbose=1,
            mode='min',
            restore_best_weights=True)
    ]
    
    # Build and sumarize model   
    with strategy.scope():
        model = unet(input_shape=(PATCHSIZE, PATCHSIZE, NUMBER_BANDS), 
                     lr=LEARNING_RATE, 
                     drop_out=DROP_OUT)
    
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

    if best_loss < 0.95:  # Adjust threshold as needed 
        
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
                dic_val = dataset_evaluation(Ypredict_val, yval, Ytrain_meta[val])
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