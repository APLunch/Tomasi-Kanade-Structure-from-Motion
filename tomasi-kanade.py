#This project is an implementation  of the Tomasi-Kanade algorithm for structure from motion
#Authors: Hongyi Fan, Yifan Yin @ Johns Hopkins University 
#Date: 2022/12/12
#Version: 1.0
#Disclaimer: This project is for educational purposes only. The authors are not responsible for any damage caused by this project. 
#This project is based on the paper "Tomasi, C., & Kanade, T. (1991). Detection and tracking of point features. Carnegie Mellon University, Pittsburgh, PA, Tech. Rep. CMU-CS-91-132."

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, os.path

# Function to get features from the image
def getFeatures(img, n=500, quality=0.01, min_distance=10, draw = False):
    #Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Use Shi-Tomasi corner detection to get the features
    features = cv2.goodFeaturesToTrack(gray, n, quality, min_distance)
    features = features.reshape(-1, 2)
    #Convert the features to float32
    features = np.float32(features)
    #If draw is true, draw the features on the image
    if draw:
        #Draw the features as red dots on the images
        for feature in features:
            x, y = feature.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        #Show the images
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return features



#main function
if __name__ == "__main__":
    #Path of the castle images
    path = "Data/castle/"
    #List all the file names in the path
    images_filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    #Sort the file names
    images_filenames.sort()
    #Get images from the file names
    images = [cv2.imread(img) for img in images_filenames]
    #Get the features from the images
    features = [getFeatures(img) for img in images] 
    #Convert the features to numpy array
    features = np.array(features)
    print(features.shape)





