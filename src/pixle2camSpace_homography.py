import cv2
import numpy as np

# ==========================
# USER SETTINGS
# ==========================

CALIBRATION_FILE = "camera_calib.yml"   # <-- path to your calibration YAML
MARKER_ID = [67, 1, 2, 3, 5, 4, 6]
MARKER_LENGTH = 0.07  # marker side length in meters (change to yours)
CAMERA_INDEX = 3

# ==========================
# SCALE: +X is short side (width). Aspect ratio is HEIGHT / WIDTH.
# ==========================
MAT_WIDTH_REAL_M = 0.205  # meters (short side)
MAT_ASPECT_RATIO = 37 / 20.5  # height / width
MAT_HEIGHT_REAL_M = MAT_WIDTH_REAL_M * MAT_ASPECT_RATIO

ARUCO_DICT = cv2.aruco.DICT_4X4_250


# ==========================
# Load Camera Calibration
# ==========================

def load_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()

    if camera_matrix is None:
        camera_matrix = fs.getNode("cameraMatrix").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("distCoeffs").mat()

    if camera_matrix is None or dist_coeffs is None:
        raise ValueError("Calibration file missing camera_matrix/distortion_coefficients (or cameraMatrix/distCoeffs).")

    return camera_matrix, dist_coeffs


# ==========================
# Create 3D Marker Points
# ==========================

def create_marker_object_points(marker_length):
    L = float(marker_length)
    return np.array(
        [
            [-L/2,  L/2, 0],
            [ L/2,  L/2, 0],
            [ L/2, -L/2, 0],
            [-L/2, -L/2, 0]
        ],
        dtype=np.float32,
    )


# ==========================
# Pose helper
# ==========================

def invert_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, t_inv


# ==========================
# Homography helpers
# ==========================

def apply_homography_to_point(H, uv):
    u, v = float(uv[0]), float(uv[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("Homography produced point at infinity (w ~ 0).")
    return np.array([q[0] / q[2], q[1] / q[2]], dtype=np.float64)


def pixel_to_Cam_Space(
    uv_undistorted,
    H,
    origin_uv_source,
    axis_p1_source,
    axis_p2_source,
    mat_width_m=MAT_WIDTH_REAL_M,
    mat_height_m=MAT_HEIGHT_REAL_M,
):
    """
    Map a pixel position in the *homography-warped* image to X/Y in meters from origin marker.

    +X is defined by axis_p1_source -> axis_p2_source (in SOURCE image), after homography.
    """
    H = np.asarray(H, dtype=np.float64)

    origin_w = apply_homography_to_point(H, origin_uv_source)
    p1_w = apply_homography_to_point(H, axis_p1_source)
    p2_w = apply_homography_to_point(H, axis_p2_source)

    v_x = (p2_w - p1_w)
    len_x_px = float(np.linalg.norm(v_x))
    if len_x_px < 1e-9:
        raise ValueError("Axis points are too close after homography; cannot define +X axis.")
    ex = v_x / len_x_px

    # Perpendicular in image coords (+u right, +v down)
    ey = np.array([ex[1], -ex[0]], dtype=np.float64)

    p_w = np.array([float(uv_undistorted[0]), float(uv_undistorted[1])], dtype=np.float64)
    d = p_w - origin_w
    dx_px = float(d.dot(ex))
    dy_px = float(d.dot(ey))

    # Scale (meters per pixel)
    m_per_px_x = float(mat_width_m) / len_x_px
    len_y_px = len_x_px * (float(mat_height_m) / float(mat_width_m))
    m_per_px_y = float(mat_height_m) / len_y_px

    x_m = dx_px * m_per_px_x
    y_m = dy_px * m_per_px_y
    return float(x_m), float(y_m)


# ============================================================
# NEW: API FOR STREAMING INTEGRATION (import-safe function calls)
# ============================================================

def get_homography_from_frame(
    frame_bgr,
    camera_matrix,
    dist_coeffs,
    marker_length=MARKER_LENGTH,
    marker_ids=MARKER_ID,
    mat_aspect_ratio=MAT_ASPECT_RATIO,
):
    """
    Detect the homography box from a SINGLE frame and compute:
      - H (source->warped)
      - origin_uv_source (marker 67 center)
      - axis points for +X (marker 3 BL -> marker 1 BR)

    Returns:
      (H, origin_uv_source, axis_p1_source, axis_p2_source)
      or (None, None, None, None) if markers not found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None, None, None

    ids = ids.flatten()
    object_points = create_marker_object_points(marker_length)

    marker_dict = {}
    for i, marker_id in enumerate(ids):
        if int(marker_id) not in marker_ids:
            continue

        image_points = corners[i].reshape(4, 2)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            continue

        marker_dict[str(int(marker_id))] = image_points.mean(axis=0)

    needed = ["3", "5", "2", "1", "67"]
    if any(k not in marker_dict for k in needed):
        return None, None, None, None

    # Your labeled points:
    # 3 = Bottom left, 5 = Top left, 2 = Top right, 1 = Bottom right, 67 = origin
    point_1 = np.array(marker_dict["3"], dtype=np.float32)   # Bottom Left
    point_2 = np.array(marker_dict["5"], dtype=np.float32)   # Top Left
    point_3 = np.array(marker_dict["2"], dtype=np.float32)   # Top Right
    point_4 = np.array(marker_dict["1"], dtype=np.float32)   # Bottom Right
    origin = np.array(marker_dict["67"], dtype=np.float32)   # origin

    # +X is RIGHT on screen: BL -> BR (short side)
    v1_2d = (point_4 - point_1).astype(np.float32)

    # Height direction is perpendicular, scaled by (height/width)
    v1 = np.append(v1_2d, 0.0)
    v2 = np.cross(v1, [0, 0, -1])
    v2 = v2 * float(mat_aspect_ratio)
    v2 = np.array(v2[:2], dtype=np.float32)

    # Source order: [BL, TL, TR, BR]
    source_points = np.array([point_1, point_2, point_3, point_4], dtype=np.float32).reshape(-1, 1, 2)

    # Dest rectangle preserving that order
    dest_points = np.array([
        point_1,                 # BL
        point_1 + v2,            # TL
        point_1 + v2 + v1_2d,    # TR
        point_1 + v1_2d          # BR
    ], dtype=np.float32).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC)
    if H is None:
        return None, None, None, None

    axis_p1 = point_1  # BL (marker 3)
    axis_p2 = point_4  # BR (marker 1)
    return H, origin, axis_p1, axis_p2


def pixel_source_to_world_xy_m(
    uv_source,
    H,
    origin_uv_source,
    axis_p1_source,
    axis_p2_source,
    mat_width_m=MAT_WIDTH_REAL_M,
    mat_height_m=MAT_HEIGHT_REAL_M,
):
    """
    (source pixel) -> (warped pixel) -> (x,y) meters from origin
    """
    uv_warped = apply_homography_to_point(H, uv_source)
    return pixel_to_Cam_Space(
        uv_undistorted=uv_warped,
        H=H,
        origin_uv_source=origin_uv_source,
        axis_p1_source=axis_p1_source,
        axis_p2_source=axis_p2_source,
        mat_width_m=mat_width_m,
        mat_height_m=mat_height_m,
    )


# ==========================
# Main (ONLY runs when executed directly)
# ==========================

if __name__ == "__main__":
    # Original behavior kept behind a guard so imports don't run camera code.
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    for _ in range(100):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        quit()

    H, origin, axis_p1, axis_p2 = get_homography_from_frame(frame, camera_matrix, dist_coeffs)
    if H is None:
        print("Could not compute homography (missing markers).")
        quit()

    undistorted = cv2.warpPerspective(frame, H, (640, 480))
    cv2.imshow("Frame", frame)
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(0)
    cap.release()