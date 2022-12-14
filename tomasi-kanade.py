#This project is an implementation  of the Tomasi-Kanade algorithm for structure from motion
#Authors: Hongyi Fan, Yifan Yin @ Johns Hopkins University 
#Date: 2022/12/12
#Version: 1.0
#Disclaimer: This project is for educational purposes only. The authors are not responsible for any damage caused by this project. 
#This project is based on the paper "Tomasi, C., & Kanade, T. (1991). Detection and tracking of point features. Carnegie Mellon University, Pittsburgh, PA, Tech. Rep. CMU-CS-91-132."

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os, os.path

# Function to get features from the image
def getFeatures(img, n=1000, quality=0.01, min_distance=3, draw = False):
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

# Function to get the good matches from the features
def getMeasurementMatrix(images):
    #Container for the good matches
    good_matches = []
    #Set the parameters for the optical flow
    lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #Get the features from the first image
    old_img = images[0]
    old_features = getFeatures(old_img)
    good_matches.append([old_features, np.ones((old_features.shape[0],1))])
    #Get the features from the images
    for img in images[1:]:
        features = getFeatures(img)
        #calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, img, old_features, None, **lk_params)
        #Select good points
        good_new = p1
        good_old = old_features
        #Old features are the new features for the next image
        old_img = img
        old_features = features
        #Append the good matches to the container
        good_matches.append([good_new,st])

    # Construct measurement matrix W
    good_index = []
    print(old_features.shape)
    for i in range(old_features.shape[0]):
        good = True
        for match in good_matches:
            if match[1][i] == 0:
                good = False
                break
        if good:
            good_index.append(i)
    selected_features = []
    for index in good_index:
        selected_features_at_frame = []
        for match in good_matches:
            selected_features_at_frame.append(match[0][index])
        selected_features.append(selected_features_at_frame)
    selected_features = np.array(selected_features)
    print(selected_features.shape)

    #Construct matrix U and V where element i,j in U is the x coordinate of the jth feature in the ith image
    #Element i,j in V is the y coordinate of the jth feature in the ith image
    U = selected_features[:, :, 0]
    V = selected_features[:, :, 1]
    #Construct the measurement matrix W which is U stacked on top of V
    print("U Shape",U.shape)
    print("V Shape",V.shape)
    W = np.vstack((U.T, V.T))
    return W

        
        

# Tomasi-Kanade Factorization from features
def factorize(measurement_matrix):
    #SVD decomposition of W
    U, S, V = np.linalg.svd(measurement_matrix,full_matrices=True)
    U = U[:, :3]
    S = S[:3]
    V = V[:3, :]
    M = U * S
    X = V
    return M,X 


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
    #Get the good matches from the images
    W = getMeasurementMatrix(images)
    print("W Shape",W.shape)
    #Factorize the measurement matrix W
    M,X = factorize(W) #M is the camera matrix, X is the 3D coordinates of the features
    print("X Shape",X.shape)
    #Plot the 3D coordinates of the features
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[0], X[1], X[2])
    plt.show()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('3D coordinates of the features')





