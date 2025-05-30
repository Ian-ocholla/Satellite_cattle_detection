#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import rasterio             # Reproject raster samples
from rasterio import windows
import fiona 
import imageio
from osgeo import gdal

from shapely.geometry import Point, MultiPoint
from shapely.geometry import mapping, shape
import numpy as np               # numerical array manipulation
import cv2
import random
from rasterio.windows import Window
from skimage import measure
from sklearn.cluster import KMeans

epsilon = 1e-07
# In[ ]:


#Now you have Ytest (true labels array), Ypredict (prediction array), meta_list (raster with crs). All are ordered.
# Next: Ytest -> points; Ypredict -> points

SEARCH_DISTANCE = 0.72 #0.72 #0.42very small increase to 0.5 i used 1.5 for THW..can be an error

def ImageToPoints(image, mask, animal_size = 25): #change from 9 for 0.4, 25px for 0.3m and 225 for 0.1m
    
    """
    Convert a binary image mask to a list of Point geometries.
    
    Args:
        image (np.ndarray): Binary image array where objects are marked.
        mask (dict): Metadata dictionary containing 'transform'.
        animal_size (int): Threshold to determine number of clusters for large regions.
    
    Returns:
        list: List of shapely.geometry.Point objects representing detected points.
    """
    #Ensure image is in uint8 format
    image = image.astype(np.uint8)
  
    # Access the transfrom from metadata dict
    transform = mask.meta['transform']
  
    cattle = []
  
    # Labels connected regioned in te binary image
    labels = measure.label(image)
    regions = measure.regionprops(labels)
    del labels #Free memory
  
  
    for region in regions:
        if region.area < 1:
            continue  # skip regions with neglible area
      
        #Replace np.int with int  
        num = int(np.ceil(region.area/animal_size)) # remove the np.
    
        if num == 1:
            # Calculate centroid of the region
            centroid = list(np.round(region.centroid))
            cattle.append(centroid) #Be aware that .append() is different from np.append()!
        else:
            clusters = KMeans(num).fit(region.coords)
            centroids = np.round(clusters.cluster_centers_)
            for centroid in centroids:
                centroid = list(centroid)
                cattle.append(centroid)

    del regions #Free memory
  
  
    points = []
    for point in cattle:
        # convert row, col to spatial coordinates
        x,y = rasterio.transform.xy(transform, point[0], point[1])
        point = Point(x, y)
        points.append(point)

    
    return points


def nearest_neighbor_within(others, point, max_distance):
    """Find nearest point among others up to a maximum distance.

    Args:
        others: a list of Points or a MultiPoint
        point: a Point
        max_distance: maximum distance to search for the nearest neighbor

    Returns:
        A shapely Point if one is within max_distance, None otherwise
    """
    search_region = point.buffer(max_distance)
    interesting_points = search_region.intersection(MultiPoint(others))

    if not interesting_points:
        closest_point = None
    elif isinstance(interesting_points, Point):
        closest_point = interesting_points
    else:
        # calculate distnaces and find the closest point
        distances = [point.distance(ip) for ip in interesting_points.geoms
                     if point.distance(ip) > 0]
        
        closest_point = interesting_points.geoms[distances.index(min(distances))]

    return closest_point 

# use the points generated from ImageToPoints(), generate the point shapefile
from osgeo import osr              ###
srs = osr.SpatialReference()       ###
srs.SetFromUserInput("EPSG:32737")  ###
wgs84 = srs.ExportToProj4() 

schema = {
    'geometry': 'Point',
    'properties': {'id': 'str'},
    }
    
    
def createShapefileObject(points, meta, wfile):
    #with fiona.open(wfile, 'w', crs=meta.get('crs').to_dict(), driver='ESRI Shapefile', schema=schema) as sink:
    with fiona.open(wfile, 'w', crs=wgs84, driver='ESRI Shapefile', schema=schema) as sink:
        for idx, point in enumerate(points):
            sink.write({
                'geometry': mapping(point),
                'properties': {'id': str(idx)},
                })
            #print(mapping(point))


def evaluation(true_points, predict_points, threshold = SEARCH_DISTANCE, index = 'cattle', ShapefilePath = None, meta = None):
    True_Positives = []
    False_Positives = []
    False_Negatives = []
    positives = predict_points.copy()
    
    for true_point in true_points:
    
        true_positive = nearest_neighbor_within(positives, true_point, threshold)
        if true_positive == None:
            False_Negatives.append(true_point)
        else:
            True_Positives.append(true_positive)
            positives.remove(true_positive)
    False_Positives = positives

    if ShapefilePath != None:
        createShapefileObject(True_Positives, meta, wfile =  os.path.join(ShapefilePath, "patch"+index+"_tp.shp"))
        createShapefileObject(False_Positives, meta, wfile = os.path.join(ShapefilePath, "patch"+index+"_fp.shp"))
        createShapefileObject(False_Negatives, meta, wfile = os.path.join(ShapefilePath, "patch"+index+"_fn.shp"))

    TP = len(True_Positives)
    FP = len(False_Positives)
    FN = len(False_Negatives)
  
    #calculate Precision
    if TP == 0 and FP == 0:
        Precision = 1
    else:
        Precision = float(TP/(TP+FP))
    

    if TP == 0 and FN == 0:
        Recall = 1
    else:
        Recall = float(TP/(TP+FN))
    
    
    F1 = 2*(Precision*Recall)/(Precision+Recall+epsilon)
    accuracy = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision":Precision,
        "Recall":Recall,
        "F1":F1
        }
    return accuracy


# In[ ]:


def dataset_evaluation(Ypredict, Ytest, meta_list, search_distance = SEARCH_DISTANCE, cluster_size = 25, point_path = None):
    Total_TP = 0
    Total_FP = 0
    Total_FN = 0
  
  
    for j in range(len(Ytest)):
        # Convert binary images to point geometries
        true_pts = ImageToPoints(Ytest[j],meta_list[j],cluster_size)
        predict_pts = ImageToPoints(Ypredict[j],meta_list[j],cluster_size)
    
        # Evaluate current image
        accuracy = evaluation(true_pts, 
            predict_pts, 
            threshold=search_distance, 
            index = j+1, 
            ShapefilePath = point_path, 
            meta = meta_list[j].meta) #change from meta_list[j].meta
        
        #aggregate metrics    
        Total_TP += accuracy['TP']
        Total_FP += accuracy['FP']
        Total_FN += accuracy['FN']
    
    # Calculate overall precision, recall and F1-score
    if (Total_TP + Total_FP) == 0:
        Total_precision = 0.0
    else:
        Total_precision = Total_TP / (Total_TP+Total_FP)
    
    if (Total_TP + Total_FN) == 0:
        Total_recall = 0.0
    else:
        Total_recall = Total_TP/(Total_TP+Total_FN)
    if (Total_precision + Total_recall) == 0:
        Total_f1 = 0.0
    else:
        Total_f1 = 2*(Total_precision*Total_recall)/(Total_precision+Total_recall+epsilon)

    #Total_precision = Total_TP/(Total_TP+Total_FP)
    #Total_recall = Total_TP/(Total_TP+Total_FN)
    #Total_f1 = 2*(Total_precision*Total_recall)/(Total_precision+Total_recall+epsilon)

    print("Cattle-level accuracy: on testing dataset  ")
    print("Total TP: ", Total_TP)
    print("Total FP: ", Total_FP)
    print("Total FN: ", Total_FN)
    print("Precision: ", Total_precision)
    print("Recall: ", Total_recall)
    print("F1-score: ", Total_f1)

    dataset_accuracy = {
        "Total_TP": Total_TP,
        "Total_FP": Total_FP,
        "Total_FN": Total_FN,
        "Precision":Total_precision,
        "Recall":Total_recall,
        "F1":Total_f1
    }
    return dataset_accuracy


# In[ ]:


def writeTiff(image, meta, wfile):
    transform = meta.meta['transform']
    crs = meta.meta['crs']
    
    new_dataset = rasterio.open(wfile, 'w', driver='GTiff',
                                height = image.shape[0], width = image.shape[1],
                                count=1, dtype=str(image.dtype),
                                crs=crs,
                                transform=transform)

    new_dataset.write(image,1)
    new_dataset.close()

