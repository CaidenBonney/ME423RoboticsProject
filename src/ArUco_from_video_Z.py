import cv2
import numpy as np
import pyrealsense2 as rs


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
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Could not open calibration file")

    camera_matrix = fs.getNode("camera_matrix").mat()
    # camera_matrix = np.array(([800, 0, 320], [0, 800, 240],[0, 0, 1]), dtype=np.float32)
    # dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    dist_coeffs = fs.getNode("distortion_coefficients").mat()

    fs.release()

    if camera_matrix is None:
        camera_matrix = fs.getNode("cameraMatrix").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("distCoeffs").mat()

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

# cap = cv2.VideoCapture(CAMERA_INDEX)

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
profile = pipeline.start(cfg)
align_to = rs.stream.color
align = rs.align(align_to)


print("Press 'q' to quit.")

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    rgb_frames = aligned_frames.get_color_frame()
    depth_frames = aligned_frames.get_depth_frame()

    rgb_image = np.asanyarray(rgb_frames.get_data())
    depth_image = np.asarray(depth_frames.get_data(), dtype=np.uint8)
    depth_data = depth_frames.get_data()
    

    if not rgb_frames or not depth_frames:
        continue

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()

        for i, marker_id in enumerate(ids):
            if marker_id == MARKER_ID:

                image_points = corners[i].reshape(4, 2)

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

                    # --- Example: convert one depth sample at pixel (u,v) into marker frame ---

                    # 1) choose pixel and get depth Z (meters)
                    
                    u,v = image_points.mean(axis=0) 

                    Z = depth_data[v,u]  # <-- you must supply this from your depth camera, in meters

                    # 2) back-project to camera coordinates
                    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
                    cx, cy = camera_matrix[0,2], camera_matrix[1,2]

                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    P_cam = np.array([[X], [Y], [Z]], dtype=np.float64)

                    # 3) pose marker->camera from solvePnP
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.reshape(3, 1)

                    # 4) transform camera -> marker
                    P_marker = R.T @ (P_cam - t)

                    x_m, y_m, z_m = P_marker.reshape(3)
                    print("Point in marker frame (m):", x_m, y_m, z_m)
                    print("Marker-frame Z (m):", z_m)

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
                    cv2.aruco.drawDetectedMarkers(rgb_image, corners, ids)
                    cv2.drawFrameAxes(rgb_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.75)

    cv2.imshow("ArUco Pose Estimation", rgb_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()