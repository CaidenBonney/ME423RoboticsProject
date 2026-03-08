"""
realsense_camera_pose_from_marker67_depth.py

RealSense + ArUco marker 67 camera pose estimation WITHOUT trusting tvec.

What it does:
- Streams RealSense color+depth, aligns depth -> color
- Detects ArUco marker 67 corners in the color image
- Gets 3D camera-frame points (meters) at those corner pixels using RealSense deprojection
- Computes marker->camera pose using:
    A) Hybrid: rotation from PnP (rvec) + translation from depth corners (NO tvec)
    B) Full 3D-3D alignment (Kabsch) from depth corners only (no rvec, no tvec), fallback

Outputs each frame (when marker found & enough depth):
- Camera position in marker frame (meters): (x, y, z)
- 4x4 transform camera->marker and marker->camera

Requires:
- pyrealsense2
- opencv-contrib-python (cv2.aruco)

Press 'q' to quit.
"""

import numpy as np
import cv2
import pyrealsense2 as rs


# ---------------- USER CONFIG ----------------
MARKER_ID = 67
MARKER_LENGTH_M = 0.07
ARUCO_DICT = cv2.aruco.DICT_4X4_250

W, H, FPS = 640, 480, 30

# Depth corner sampling: try a small neighborhood to reduce missing depth
NEIGHBOR_RADIUS_PX = 2  # 0 = sample exactly at corner pixel
MIN_VALID_CORNERS = 3   # need at least 3 to estimate pose robustly


# ---------------- MARKER MODEL ----------------
def create_marker_object_points(marker_length_m: float) -> np.ndarray:
    """4 corners in marker frame (meters), centered at marker center, on Z=0 plane."""
    L = marker_length_m
    return np.array(
        [
            [-L / 2,  L / 2, 0],
            [ L / 2,  L / 2, 0],
            [ L / 2, -L / 2, 0],
            [-L / 2, -L / 2, 0],
        ],
        dtype=np.float64
    )


# ---------------- UTILS ----------------
def build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3,)
    return T

def robust_depth_at_pixel(depth_frame, u: int, v: int, radius: int) -> float:
    """
    Get a more reliable depth (meters) by searching a small neighborhood
    for the nearest valid (non-zero) depth.
    """
    if radius <= 0:
        return float(depth_frame.get_distance(u, v))

    best = 0.0
    # Prefer closest valid sample (smallest positive depth) or first valid; here choose median of valid
    vals = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            z = float(depth_frame.get_distance(u + du, v + dv))
            if z > 0:
                vals.append(z)
    if not vals:
        return 0.0
    return float(np.median(vals))

def deproject(u: int, v: int, depth_m: float, intr: rs.intrinsics) -> np.ndarray:
    """RealSense deprojection returns [X,Y,Z] meters in the camera optical frame."""
    p = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
    return np.array(p, dtype=np.float64)

def estimate_translation_from_depth_corners(R_cam_from_marker: np.ndarray,
                                            corners_uv: np.ndarray,
                                            obj_pts_marker: np.ndarray,
                                            depth_frame,
                                            intr: rs.intrinsics,
                                            radius_px: int) -> np.ndarray | None:
    """
    Given R (marker->camera), estimate translation t (marker->camera) from depth at corners:
        P_cam_i ≈ R * P_obj_i + t
        t_i = P_cam_i - R*P_obj_i
    Return robust median of t_i across valid corners.
    """
    t_list = []
    for (u, v), P_obj in zip(corners_uv, obj_pts_marker):
        ui, vi = int(round(u)), int(round(v))
        z = robust_depth_at_pixel(depth_frame, ui, vi, radius_px)
        if z <= 0:
            continue
        P_cam = deproject(ui, vi, z, intr)  # meters
        t_i = P_cam - (R_cam_from_marker @ P_obj)
        t_list.append(t_i)

    if len(t_list) < MIN_VALID_CORNERS:
        return None

    t_stack = np.vstack(t_list)
    return np.median(t_stack, axis=0)

def kabsch_rigid_transform(A: np.ndarray, B: np.ndarray):
    """
    Compute rigid transform (R,t) such that:
        B ≈ R*A + t
    A: (N,3) points in marker frame
    B: (N,3) corresponding points in camera frame
    Returns (R,t).
    """
    assert A.shape == B.shape and A.shape[1] == 3
    N = A.shape[0]
    if N < 3:
        return None, None

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection if needed
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t

def detect_marker_corners(frame_bgr: np.ndarray, marker_id: int, dictionary, params):
    """Returns (found, corners_uv, all_corners, ids). corners_uv is (4,2) for marker_id."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    all_corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return False, None, all_corners, ids

    ids_flat = ids.flatten()
    for i, mid in enumerate(ids_flat):
        if int(mid) == int(marker_id):
            corners_uv = all_corners[i].reshape(4, 2).astype(np.float64)
            return True, corners_uv, all_corners, ids

    return False, None, all_corners, ids


# ---------------- MAIN ----------------
def main():
    # RealSense setup
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    align = rs.align(rs.stream.color)

    # Grab RealSense intrinsics directly from the running stream (important!)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    # OpenCV camera matrix from intrinsics (for drawing axes / optional PnP rotation)
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)  # often fine in practice for RealSense streams

    # ArUco setup
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    obj_pts_marker = create_marker_object_points(MARKER_LENGTH_M)  # (4,3)

    cv2.namedWindow("pose", cv2.WINDOW_AUTOSIZE)

    print("Running. Looking for marker 67. Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            frame = np.asanyarray(color.get_data())
            vis = frame.copy()

            found, corners_uv, all_corners, ids = detect_marker_corners(frame, MARKER_ID, dictionary, params)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, all_corners, ids)

            if not found:
                cv2.putText(vis, "Marker 67 not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("pose", vis)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue

            # --- Get 3D camera points at corners from depth (meters) ---
            P_cam_list = []
            P_obj_list = []
            for (u, v), P_obj in zip(corners_uv, obj_pts_marker):
                ui, vi = int(round(u)), int(round(v))
                z = robust_depth_at_pixel(depth, ui, vi, NEIGHBOR_RADIUS_PX)
                if z <= 0:
                    continue
                P_cam = deproject(ui, vi, z, intr)
                P_cam_list.append(P_cam)
                P_obj_list.append(P_obj)

            P_cam_arr = np.array(P_cam_list, dtype=np.float64)  # (N,3)
            P_obj_arr = np.array(P_obj_list, dtype=np.float64)  # (N,3)

            if P_cam_arr.shape[0] < MIN_VALID_CORNERS:
                cv2.putText(vis, "Marker found but not enough valid depth at corners", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("pose", vis)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue

            # -------------------------
            # A) Hybrid pose: R from PnP, t from depth corners
            # -------------------------
            # Compute rvec via PnP ONLY for rotation (tvec will be ignored)
            # Use the full 4-corner 2D set even if some depth missing; that's ok for rotation.
            ok_pnp, rvec, tvec = cv2.solvePnP(
                obj_pts_marker.astype(np.float32),
                corners_uv.astype(np.float32),
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            use_method = ""
            if ok_pnp:
                R_cam_from_marker, _ = cv2.Rodrigues(rvec)
                t_from_depth = estimate_translation_from_depth_corners(
                    R_cam_from_marker, corners_uv, obj_pts_marker, depth, intr, NEIGHBOR_RADIUS_PX
                )
                if t_from_depth is not None:
                    R = R_cam_from_marker
                    t = t_from_depth
                    use_method = "Hybrid: R from PnP, t from depth corners"
                else:
                    R = None
                    t = None
            else:
                R = None
                t = None

            # -------------------------
            # B) Fallback: Full 3D-3D alignment from depth corners
            # -------------------------
            if R is None or t is None:
                R_k, t_k = kabsch_rigid_transform(P_obj_arr, P_cam_arr)
                if R_k is None:
                    cv2.putText(vis, "Pose failed (not enough depth corners)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("pose", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                    continue
                R, t = R_k, t_k
                use_method = "Kabsch: full pose from depth corners"

            # Now we have marker->camera pose: P_cam = R*P_marker + t
            T_cam_from_marker = build_T(R, t)

            # Camera pose in marker frame (inverse)
            R_marker_from_cam = R.T
            t_marker_from_cam = -R_marker_from_cam @ t
            T_marker_from_cam = build_T(R_marker_from_cam, t_marker_from_cam)

            x, y, z = t_marker_from_cam.tolist()

            # Draw axes for visualization using the pose (need rvec,tvec form)
            rvec_draw, _ = cv2.Rodrigues(R)
            tvec_draw = t.reshape(3, 1)
            cv2.drawFrameAxes(vis, K, dist, rvec_draw, tvec_draw, MARKER_LENGTH_M * 0.75)

            cv2.putText(vis, use_method, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis, f"Camera in marker67 frame (m): x={x:.3f}, y={y:.3f}, z={z:.3f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Optional debug prints
            # print("T_marker_from_cam:\n", T_marker_from_cam)

            cv2.imshow("pose", vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()