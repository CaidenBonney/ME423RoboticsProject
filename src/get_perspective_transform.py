import cv2
import numpy as np

# ==========================
# USER SETTINGS
# ==========================

CALIBRATION_FILE = "camera_calib.yml"   # <-- path to your calibration YAML
MARKER_ID = 67
MARKER_LENGTH = 0.07  # marker side length in meters (change to yours)
CAMERA_INDEX = 3

# IMPORTANT: must match how marker 67 was created
ARUCO_DICT = cv2.aruco.DICT_4X4_250

# ==========================
# Load Camera Calibration
# ==========================

def load_calibration(path):
    # fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    # if not fs.isOpened():
    #     raise FileNotFoundError("Could not open calibration file")

    # camera_matrix = fs.getNode("camera_matrix").mat()
    camera_matrix = np.array(([800, 0, 320], [0, 800, 240],[0, 0, 1]), dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # dist_coeffs = fs.getNode("distortion_coefficients").mat()

    # fs.release()

    # if camera_matrix is None:
    #     camera_matrix = fs.getNode("cameraMatrix").mat()
    # if dist_coeffs is None:
    #     dist_coeffs = fs.getNode("distCoeffs").mat()

    return camera_matrix, dist_coeffs


# ==========================
# Create 3D Marker Points
# ==========================

def create_marker_object_points(marker_length):
    L = marker_length
    return np.array([
        [-L/2,  L/2, 0],
        [ L/2,  L/2, 0],
        [ L/2, -L/2, 0],
        [-L/2, -L/2, 0]
    ], dtype=np.float32)


# ==========================
# Invert Pose (Marker->Camera  →  Camera->Marker)
# ==========================

def invert_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, t_inv


# ==========================
# Main
# ==========================

camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)
object_points = create_marker_object_points(MARKER_LENGTH)

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(CAMERA_INDEX)
for i in range(100):  # warm up camera by reading a few frames
    r,f = cap.read()

print("Press 'q' to quit.")

marker_centers = []
ret, frame = cap.read()
if not ret:
    quit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect markers
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None:
    ids = ids.flatten()

    for i, marker_id in enumerate(ids):
        if marker_id == MARKER_ID:

            image_points = corners[i].reshape(4, 2)
            # print(image_points)
            # Estimate marker pose w.r.t camera
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Invert to get camera pose w.r.t marker
                rvec_cam, tvec_cam = invert_pose(rvec, tvec)

                R_cam, _ = cv2.Rodrigues(rvec_cam)

                T_cam_in_marker = np.eye(4)
                T_cam_in_marker[:3, :3] = R_cam
                T_cam_in_marker[:3, 3] = tvec_cam.reshape(3)

                """
                print("\n=== Marker 67 Detected ===")
                print("Camera Position in Marker Frame (meters):")
                print(tvec_cam.reshape(3))
                print("Camera Rotation Vector (Rodrigues, radians):")
                print(rvec_cam.reshape(3))
                print("4x4 Transformation Matrix:")
                print(T_cam_in_marker)
                """
                # Draw results for visualization
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.75)
                marker_centers.append(image_points.mean(axis=0))  # (u,v) center of the marker in pixel coordinates
                # print(marker_centers)

x_spots = []
y_spots = []
for center in marker_centers:
    marker_center = (int(center[0]), int(center[1]))
    x_spots.append(center[0])
    y_spots.append(center[1])
    cv2.circle(frame, center=marker_center, radius=50, color=(0, 255, 0), thickness=2)
true_centers = (int(np.mean(x_spots)), int(np.mean(y_spots)))


cv2.circle(frame, center=true_centers, radius=50, color=(255, 0, 0), thickness=2)
cv2.imshow("ArUco Pose Estimation", frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

# calculate destination points for homography (normal view of the marker)
marker_centers = np.array(marker_centers) 
hull = cv2.convexHull(marker_centers)
print("hull ", hull)

# define edges
edge_1_len = np.linalg.norm(hull[0] - hull[1])
edge_2_len = np.linalg.norm(hull[1] - hull[2])
print("edge_1 ", edge_1_len)
print("edge_2 ", edge_2_len)

if edge_1_len > edge_2_len:
    sequenced = [hull[0], hull[1], hull[2], hull[3]]
else:
    sequenced = [hull[1], hull[2], hull[3], hull[0]]

mat_aspect_ratio = 37/25.5
 # the longest edge should be the height of the marker, so we can calculate the height using the aspect ratio


# calculate destination points for homography (normal view of the marker)

# destination points for homography (normal view of the marker), should be calculate to match shape of the true mat
homography, _ = cv2.findHomography(marker_centers, dest_points)
undistorted = cv2.warpPerspective(frame, homography, (640, 480))
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)