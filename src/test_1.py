# ## License: Apache 2.0. See LICENSE file in root directory.
# ## Copyright(c) 2015-2017 RealSense, Inc. All Rights Reserved.

# #####################################################
# ## librealsense tutorial #1 - Accessing depth data ##
# #####################################################

# # First import the library
import pyrealsense2 as rs
import cv2
import numpy as np
# import os
# print(os.listdir())
 
# Read image
im = cv2.imread("pictures/ball_2.jpg", cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Detect bright blobs (white ball)
params.filterByColor = True
params.blobColor = 255

# Size filtering (tune minArea for your image)
params.filterByArea = True
params.minArea = 500 # used to be 100
params.maxArea = 200000

# Shape filtering (reject edges/corners)
params.filterByCircularity = True
params.minCircularity = 0.7 # larger number because ping pong ball are round ash

params.filterByInertia = True # used to be false
params.minInertiaRatio = 0.5

params.filterByConvexity = True # used to be false
params.minConvexity = 0.8

detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)