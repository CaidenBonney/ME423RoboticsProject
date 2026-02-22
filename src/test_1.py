# Standard imports
import cv2
import numpy as np
import time
# import os
# print(os.listdir())
 
# Read image
im = cv2.imread("pictures/ball_2.jpg", cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)
 
start = time.time()
# Detect blobs.
keypoints = detector.detect(im)
end = time.time() 
print(f'detection time: {end - start}')
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("detected.png", im_with_keypoints)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)