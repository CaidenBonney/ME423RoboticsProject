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
im = cv2.imread("pictures/ball_3.jpg", cv2.IMREAD_GRAYSCALE)

# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Detect bright blobs (white ball)
params.filterByColor = True
params.blobColor = 255

# Size filtering (tune minArea for your image)
params.filterByArea = True
params.minArea = 1000 # used to be 100
params.maxArea = 200_000

# Shape filtering (reject edges/corners)
params.filterByCircularity = True
params.minCircularity = 0.6 # larger number because ping pong ball are round ash

params.filterByInertia = True # used to be false
params.minInertiaRatio = 0.1

params.filterByConvexity = True # used to be false
params.minConvexity = 0.5

detector = cv2.SimpleBlobDetector_create(params)
 
blur = cv2.GaussianBlur(im, (9,9), 0)

# adaptive threshold handles uneven lighting / shadows
bw = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    101,  # blockSize (odd) -> bigger = smoother lighting model
    -15   # C -> shifts threshold; make more negative to include darker parts
)

# fill in shadow gaps / holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

# Detect blobs.
keypoints = detector.detect(bw)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("bw", bw)
cv2.waitKey(0)