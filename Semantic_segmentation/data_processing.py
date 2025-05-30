# Import necessary libraries

import osgeo
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import rasterio
from rasterio import windows

import fiona
from fiona.crs import from_epsg
from shapely.geometry import Point, MultiPoint, mapping, shape

#Image prcoessing and visualization
import imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import exposure

# Machine learning and Tensorflow
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta
from tensorflow.keras import regularizers

#clustering
from sklearn.cluster import KMeans

#system and helper libraries
import os
import random
import numpy as np
import gc
import pandas as pd

# Set the working folder
Folder = "/scratch/project_2008354/Cattle_UNet"
Data_folder = os.path.join(Folder, "SampleData")

# Add the core folder to system path
import sys
sys.path.insert(0, os.path.join(Folder, "modules"))

# Import custom modules
import data_generator
from data_generator import DataGenerator

import data_augmentation
from data_augmentation import *

import counting
from counting import *

import visualization
from visualization import *

# Define constants
PATCHSIZE = 336 #336
READ_CHANNEL = [1, 2, 3]  # Use RGB bands
NUMBER_BANDS = len(READ_CHANNEL)

# Helper function to load and process data
def load_data(data_path, year, load_train=True): # inclusion of load_train
    IMAGE_PATH = os.path.join(data_path, f"data_{year}/3_Train_test/train/image")
    LABEL_PATH = os.path.join(data_path, f"data_{year}/3_Train_test/train/mask")
    TEST_IMAGE_PATH = os.path.join(data_path, f"data_{year}/3_Train_test/test/image")
    TEST_LABEL_PATH = os.path.join(data_path, f"data_{year}/3_Train_test/test/mask")
    
    # Load the training and testing datasets
    Data = DataGenerator(IMAGE_PATH, LABEL_PATH, TEST_IMAGE_PATH, TEST_LABEL_PATH, input_image_channel=READ_CHANNEL, patchsize=PATCHSIZE)
    
    # Generate patches
    patches = Data.generate_patches()
    
    # Conditional loading
    if load_train:
        # Load both train and test data
        Xtrain = patches['Xtrain']
        Ytrain = patches['Ytrain']
    else:
        # Return placeholders for train data
        Xtrain, Ytrain = None, None
        
        # Xtest = patches['Xtest']
        # Ytest = patches['Ytest']
    
    # Convert test data type to float (required by U-Net) --always load test data
    Xtest = np.float32(patches['Xtest'])
    Ytest = np.float32(patches['Ytest'])
    
    #Xtest = np.float32(Xtest)
    #Ytest = np.float32(Ytest)
    
    # Return data and metadata
    return Xtrain, Ytrain, Xtest, Ytest, Data.train_meta_list, Data.test_meta_list

# Load data for multiple years
#years = ['Choke_343_2022Jan', 'Lumo_901_2022Jan', 'Lumo_729_2022Jan', 'Lumo_305_2022Jan', 'Lumo_WV3_2022Jan'] #increase or reduce
years = ['Lumo_PNEO3_2022Jan']
Xtrain_list, Ytrain_list, Xtest_list, Ytest_list = [], [], [], []
Ytrain_meta_list, Ytest_meta_list = [], []

for year in years:
    Xtrain, Ytrain, Xtest, Ytest, Ytrain_meta, Ytest_meta = load_data(Data_folder, year)
    Xtrain_list.append(Xtrain)
    Ytrain_list.append(Ytrain)
    Xtest_list.append(Xtest)
    Ytest_list.append(Ytest)
    Ytrain_meta_list.append(Ytrain_meta)
    Ytest_meta_list.append(Ytest_meta)

# Concatenate datasets
Xtrain = np.concatenate(Xtrain_list)
Ytrain = np.concatenate(Ytrain_list)
Xtest = np.concatenate(Xtest_list)
Ytest = np.concatenate(Ytest_list)

Ytrain_meta = np.concatenate(Ytrain_meta_list)
Ytest_meta = np.concatenate(Ytest_meta_list)

print(np.shape(Xtrain))
print(np.shape(Ytrain))
print(np.shape(Xtest))
print(np.shape(Ytest))

print(np.shape(Ytrain_meta))
print(np.shape(Ytest_meta))

#Check the percentage of cattle pixels with respect to non-wildebeest
c_px=len(Ytrain[Ytrain==1])/(len(Ytrain[Ytrain==1])+len(Ytrain[Ytrain==0]))
nonc_px=len(Ytrain[Ytrain==0])/(len(Ytrain[Ytrain==1])+len(Ytrain[Ytrain==0]))
print("Pixel percentage of cattle = " + str(c_px* 100)) #We can observe that the percentage of cattle pixels are really less 
print("Pixel percentage of non-cattle = " + str(nonc_px* 100))