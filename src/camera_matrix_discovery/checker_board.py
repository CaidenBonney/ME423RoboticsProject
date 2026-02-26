import cv2
import numpy as np
import glob

# Checkerboard settings
CHECKERBOARD = (6, 6)
SQUARE_SIZE = 0.0255  # meters (2.55 cm squares)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []
images = glob.glob('sample_images/*.png')  # Put images in folder
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    cv2.imshow('img', img)
    cv2.waitKey(2000)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        # cv2.imshow('img', img)
        cv2.waitKey(200)