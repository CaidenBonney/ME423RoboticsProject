import numpy as np
import cv2

CALIBRATION_FILE = "camera_calib.yml"   # <-- path to your calibration YAML

def pixel_to_camera_3d(u, v, Z, K, dist):
    """
    u, v: pixel coordinates (ball center)
    Z: planar distance (depth along camera Z axis), in meters (or same unit you want out)
    K: 3x3 camera matrix
    dist: distortion coefficients (from calibration)
    
    Returns: (X, Y, Z) in camera coordinates.
    OpenCV camera coords: +X right, +Y down, +Z forward.
    """
    # Input point shape must be (N,1,2)
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)

    # Undistort to normalized image coordinates (x_n, y_n)
    undist = cv2.undistortPoints(pts, K, dist)  # returns (N,1,2) normalized
    x_n, y_n = undist[0, 0, 0], undist[0, 0, 1]

    X = x_n * Z
    Y = y_n * Z
    return np.array([X, Y, Z], dtype=np.float64)

def load_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Could not open calibration file")

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()

    fs.release()

    if camera_matrix is None:
        camera_matrix = fs.getNode("cameraMatrix").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("distCoeffs").mat()

    return camera_matrix, dist_coeffs

# Example:

camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

u, v = 420, 240          # pixel center of the ball
Z = 1.75                 # meters (depth along optical axis)

P_cam = pixel_to_camera_3d(u, v, Z, camera_matrix, dist_coeffs)
print("3D position in camera frame:", P_cam)