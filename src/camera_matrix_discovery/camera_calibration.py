import cv2
import numpy as np
import glob
import os


# Checkerboard settings
CHECKERBOARD = (6, 6)
SQUARE_SIZE = 0.0255  # meters (2.55 cm squares)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []


script_dir = os.path.dirname(os.path.abspath(__file__))
images = glob.glob(os.path.join(script_dir, "sample_images", "*.png"))

size = [640, 480]  # Image size (width, height) - change if your images are different
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)


ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
objpoints,
imgpoints,
size,
None,
None)

# Save calibration file
fs = cv2.FileStorage("camera_calib.yml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("distortion_coefficients", dist_coeffs)
fs.release()

print("Calibration saved to camera_calib.yml")