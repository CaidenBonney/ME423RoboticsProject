"""
rgbd_homography_pnp_xyz.py

All-in-one script that:
1) Streams aligned RGB + depth from an Intel RealSense depth camera
2) Computes a homography once (using your pixle2camSpace_homography helper)
3) Tracks a white moving object (your “ball” style detector)
4) Estimates ArUco marker pose (solvePnP) for MARKER_ID
5) Produces:
   - Planar (X,Y) in meters from homography
   - Full 3D (X,Y,Z) in meters in the ArUco marker frame from depth + PnP

Notes:
- Homography is used ONLY for planar XY mapping on the mat.
- Depth->3D uses ORIGINAL image pixels (u,v), not homography-warped pixels.
- This assumes your depth is aligned to the color stream (we do rs.align(color)).

Dependencies:
- pyrealsense2
- opencv-contrib-python (for cv2.aruco)
- your module: pixle2camSpace_homography.py (same interface you used before)

Press 'q' to quit.
"""

import time
import math
import numpy as np
import cv2
import pyrealsense2 as rs

# ==========================
# Imports from YOUR homography helper module (as in your homography script)
# ==========================
from pixle2camSpace_homography import (
    load_calibration,
    get_homography_from_frame,
    pixel_source_to_world_xy_m,
    CALIBRATION_FILE,
    MAT_WIDTH_REAL_M,
    MAT_HEIGHT_REAL_M,
)

# ==========================
# USER SETTINGS (from your ArUco script + sensible defaults)
# ==========================
MARKER_ID = 67
MARKER_LENGTH_M = 0.07  # meters
ARUCO_DICT = cv2.aruco.DICT_4X4_250

# RealSense stream settings
W, H = 640, 480
FPS = 60

# Warmup frames for background subtractor
WARMUP_FRAMES = 30

# ==========================
# Helper functions
# ==========================
def create_marker_object_points(marker_length_m: float) -> np.ndarray:
    """Marker corners in marker frame (meters), centered at marker center, lying on Z=0 plane."""
    L = marker_length_m
    return np.array(
        [
            [-L / 2,  L / 2, 0],
            [ L / 2,  L / 2, 0],
            [ L / 2, -L / 2, 0],
            [-L / 2, -L / 2, 0],
        ],
        dtype=np.float32
    )

def backproject_pixel_to_cam(u: int, v: int, depth_m: float, K: np.ndarray) -> np.ndarray:
    """
    Back-project a pixel + depth (meters) to a 3D point in the *camera* coordinate frame.
    Assumes depth_m is Z along the optical axis (RealSense aligned depth commonly behaves this way).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float64)

def cam_point_to_marker(P_cam: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    solvePnP gives marker->camera: P_cam = R*P_marker + t
    So camera->marker: P_marker = R^T*(P_cam - t)
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,)
    return R.T @ (P_cam - t)

def detect_marker_pose(frame_bgr: np.ndarray,
                       marker_id: int,
                       object_points: np.ndarray,
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       dictionary,
                       detector_params):
    """
    Detect ArUco markers and return (success, rvec, tvec, corners, ids).
    Uses SOLVEPNP_IPPE_SQUARE (good for planar square markers).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return False, None, None, corners, ids

    ids_flat = ids.flatten()
    for i, mid in enumerate(ids_flat):
        if int(mid) == int(marker_id):
            image_points = corners[i].reshape(4, 2).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            return bool(success), rvec, tvec, corners, ids

    return False, None, None, corners, ids

# ==========================
# Main
# ==========================
def main():
    print("Starting...")

    # --- Load camera intrinsics/distortion (same calibration used for homography + ArUco) ---
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

    # --- ArUco settings ---
    object_points = create_marker_object_points(MARKER_LENGTH_M)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()

    # --- RealSense pipeline ---
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    profile = pipeline.start(cfg)

    # Align depth to color
    align = rs.align(rs.stream.color)

    # --- Background subtractor for motion filtering ---
    bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    # Warm up background model
    print("Warming up background model...")
    for _ in range(WARMUP_FRAMES):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        if not color:
            continue
        frame_bgr = np.asanyarray(color.get_data())
        bs.apply(frame_bgr, learningRate=0.05)
    print("Background model warmed up.")

    # --- One-time homography calibration (as in your homography script) ---
    homography = None
    origin_uv = None
    axis_p1 = None
    axis_p2 = None

    print("Searching for homography box (AruCo markers)...")
    while homography is None:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            continue

        frame_bgr = np.asanyarray(color.get_data())

        H_tmp, origin_tmp, axis_p1_tmp, axis_p2_tmp = get_homography_from_frame(
            frame_bgr=frame_bgr,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if H_tmp is not None:
            homography = H_tmp
            origin_uv = origin_tmp
            axis_p1 = axis_p1_tmp
            axis_p2 = axis_p2_tmp
            print("HOMOGRAPHY CALIBRATION COMPLETE.")
            break

    # --- Object detection thresholds (from your homography script) ---
    S_high = 70
    V_low = 185

    area_min = 80
    area_max = 8000
    circularity_min = 0.25
    solidity_min = 0.40
    aspect_max = 1.8

    k5 = np.ones((5, 5), np.uint8)
    k3 = np.ones((3, 3), np.uint8)

    last_pts = []

    cv2.namedWindow("rgb_cam", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("depth_cam", cv2.WINDOW_AUTOSIZE)

    print("Running. Press 'q' to quit.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            continue

        frame_bgr = np.asanyarray(color.get_data())
        out = frame_bgr.copy()

        # ---- Depth visualization only (do NOT use this for measurement) ----
        depth_vis = np.asanyarray(depth.get_data())  # uint16
        depth_vis_8u = cv2.convertScaleAbs(depth_vis, alpha=0.03)
        depth_map = cv2.applyColorMap(depth_vis_8u, cv2.COLORMAP_JET)

        # ==========================
        # 1) Detect ArUco pose (PnP)
        # ==========================
        pnp_ok, rvec, tvec, corners, ids = detect_marker_pose(
            frame_bgr=frame_bgr,
            marker_id=MARKER_ID,
            object_points=object_points,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            dictionary=dictionary,
            detector_params=detector_params,
        )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)

        if pnp_ok:
            cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M * 0.75)
        else:
            cv2.putText(out, "PnP: marker not found", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ==========================
        # 2) Detect moving white object center (u,v) in the ORIGINAL image
        # ==========================
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, (0, 0, V_low), (179, S_high, 255))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k5, iterations=1)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k5, iterations=2)

        fg = bs.apply(frame_bgr, learningRate=0.002)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
        fg = cv2.dilate(fg, k5, iterations=1)

        mask = cv2.bitwise_and(white, fg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        pred = None
        if len(last_pts) >= 2:
            (x1, y1), (x2, y2) = last_pts[-2], last_pts[-1]
            pred = (x2 + (x2 - x1), y2 + (y2 - y1))
        elif len(last_pts) == 1:
            pred = last_pts[-1]

        for c in contours:
            area = cv2.contourArea(c)
            if area < area_min or area > area_max:
                continue

            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue

            circularity = 4 * math.pi * area / (peri * peri)

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = (area / hull_area) if hull_area > 0 else 0.0

            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w / max(1, h), h / max(1, w))

            if circularity < circularity_min:
                continue
            if solidity < solidity_min:
                continue
            if aspect > aspect_max:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            dist_pred = 0.0
            if pred is not None:
                dist_pred = math.hypot(cx - pred[0], cy - pred[1])

            score = (
                -2.0 * dist_pred
                + 0.25 * area
                + 350.0 * circularity
                + 250.0 * solidity
                - 60.0 * aspect
            )

            if best is None or score > best["score"]:
                best = dict(
                    score=score,
                    cx=cx, cy=cy,
                    area=area,
                    circularity=circularity,
                    solidity=solidity,
                    aspect=aspect,
                    hull=hull,
                    bbox=(x, y, w, h),
                )

        # ==========================
        # 3) If we found the object center: compute homography XY and PnP+depth XYZ
        # ==========================
        if best is not None:
            cx, cy = best["cx"], best["cy"]
            u, v = int(round(cx)), int(round(cy))

            last_pts.append((cx, cy))
            if len(last_pts) > 10:
                last_pts = last_pts[-10:]

            cv2.drawContours(out, [best["hull"]], -1, (0, 255, 0), 2)
            cv2.circle(out, (u, v), 4, (255, 0, 0), -1)

            x, y, w, h = best["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # ---- Homography planar XY (meters on your mat plane) ----
            try:
                x_plane_m, y_plane_m = pixel_source_to_world_xy_m(
                    uv_source=(cx, cy),
                    H=homography,
                    origin_uv_source=origin_uv,
                    axis_p1_source=axis_p1,
                    axis_p2_source=axis_p2,
                    mat_width_m=MAT_WIDTH_REAL_M,
                    mat_height_m=MAT_HEIGHT_REAL_M,
                )
                cv2.putText(out, f"Plane XY (m): x={x_plane_m:.3f}, y={y_plane_m:.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception:
                x_plane_m, y_plane_m = None, None
                cv2.putText(out, "Plane XY: map failed",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ---- Depth at (u,v) in meters ----
            depth_m = float(depth.get_distance(u, v))
            if depth_m <= 0:
                cv2.putText(out, "Depth: invalid",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(out, f"Depth (m): {depth_m:.3f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ---- Full 3D point in marker frame (meters) using PnP + depth ----
                if pnp_ok:
                    P_cam = backproject_pixel_to_cam(u, v, depth_m, camera_matrix)
                    P_marker = cam_point_to_marker(P_cam, rvec, tvec)
                    xM, yM, zM = float(P_marker[0]), float(P_marker[1]), float(P_marker[2])

                    cv2.putText(out, f"Marker XYZ (m): x={xM:.3f}, y={yM:.3f}, z={zM:.3f}",
                                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(out, "Marker XYZ: need marker pose",
                                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---- Show windows ----
        cv2.imshow("rgb_cam", out)
        cv2.imshow("depth_cam", depth_map)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()