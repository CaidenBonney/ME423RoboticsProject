import math
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------------- USER CONFIG ----------------
MARKER_ID = 67
MARKER_LENGTH_M = 0.07
ARUCO_DICT = cv2.aruco.DICT_4X4_250

W, H, FPS = 640, 480, 30

NEIGHBOR_RADIUS_PX = 2     # depth sampling neighborhood for marker corners
BALL_DEPTH_RADIUS_PX = 2   # depth sampling neighborhood for ball center
MIN_VALID_CORNERS = 3

# Ball detector thresholds (tune if needed)
S_HIGH = 70
V_LOW = 185
AREA_MIN = 80
AREA_MAX = 12000
CIRC_MIN = 0.25
SOLID_MIN = 0.40
ASPECT_MAX = 1.8

WARMUP_FRAMES = 30


# ---------------- TEXT OVERLAY (wrap to screen) ----------------
def draw_overlay(img, lines, x=10, y=25, font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.6, thickness=2, color=(0, 255, 0),
                 line_gap=6, margin=10):
    h, w = img.shape[:2]

    def get_w_h(s):
        (tw, th), _ = cv2.getTextSize(s, font, font_scale, thickness)
        return tw, th

    cur_y = y
    for line in lines:
        if line is None:
            continue
        s = str(line)

        while s:
            tw, th = get_w_h(s)
            if x + tw <= w - margin:
                cv2.putText(img, s, (x, cur_y), font, font_scale, color, thickness, cv2.LINE_AA)
                cur_y += th + line_gap
                break

            split_idx = -1
            for i in range(len(s)):
                chunk = s[:i+1]
                twc, _ = get_w_h(chunk)
                if x + twc > w - margin:
                    break
                if s[i].isspace():
                    split_idx = i

            if split_idx <= 0:
                max_i = 1
                for i in range(1, len(s)+1):
                    twc, th = get_w_h(s[:i])
                    if x + twc > w - margin:
                        break
                    max_i = i
                head = s[:max_i]
                tail = s[max_i:]
            else:
                head = s[:split_idx].rstrip()
                tail = s[split_idx+1:].lstrip()

            cv2.putText(img, head, (x, cur_y), font, font_scale, color, thickness, cv2.LINE_AA)
            cur_y += th + line_gap
            s = tail

        if cur_y > h - margin:
            break


# ---------------- MARKER MODEL ----------------
def create_marker_object_points(marker_length_m: float) -> np.ndarray:
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
def robust_depth_at_pixel(depth_frame, u: int, v: int, radius: int) -> float:
    if radius <= 0:
        return float(depth_frame.get_distance(u, v))
    vals = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            z = float(depth_frame.get_distance(uu, vv))
            if z > 0:
                vals.append(z)
    if not vals:
        return 0.0
    return float(np.median(vals))

def deproject(u: int, v: int, depth_m: float, intr: rs.intrinsics) -> np.ndarray:
    p = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
    return np.array(p, dtype=np.float64)  # [X,Y,Z] meters in camera frame

def build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3,)
    return T

def kabsch_rigid_transform(A: np.ndarray, B: np.ndarray):
    # B ≈ R*A + t
    if A.shape[0] < 3:
        return None, None
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    Hm = AA.T @ BB
    U, S, Vt = np.linalg.svd(Hm)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

def detect_marker_corners(frame_bgr: np.ndarray, marker_id: int, dictionary, params):
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

def estimate_translation_from_depth_corners(R_cam_from_marker: np.ndarray,
                                            corners_uv: np.ndarray,
                                            obj_pts_marker: np.ndarray,
                                            depth_frame,
                                            intr: rs.intrinsics,
                                            radius_px: int) -> np.ndarray | None:
    t_list = []
    for (u, v), P_obj in zip(corners_uv, obj_pts_marker):
        ui, vi = int(round(u)), int(round(v))
        z = robust_depth_at_pixel(depth_frame, ui, vi, radius_px)
        if z <= 0:
            continue
        P_cam = deproject(ui, vi, z, intr)
        t_i = P_cam - (R_cam_from_marker @ P_obj)
        t_list.append(t_i)

    if len(t_list) < MIN_VALID_CORNERS:
        return None

    t_stack = np.vstack(t_list)
    return np.median(t_stack, axis=0)

def get_camera_to_marker_transform(frame_bgr, depth_frame, intr, K, dist,
                                  dictionary, params, obj_pts_marker):
    """
    Returns (ok, R_marker_from_cam, t_marker_from_cam, debug_method_string, corners, ids)
    """
    found, corners_uv, all_corners, ids = detect_marker_corners(frame_bgr, MARKER_ID, dictionary, params)
    if not found:
        return False, None, None, "Marker not found", all_corners, ids

    # gather 3D corner points from depth
    P_cam_list = []
    P_obj_list = []
    for (u, v), P_obj in zip(corners_uv, obj_pts_marker):
        ui, vi = int(round(u)), int(round(v))
        z = robust_depth_at_pixel(depth_frame, ui, vi, NEIGHBOR_RADIUS_PX)
        if z <= 0:
            continue
        P_cam_list.append(deproject(ui, vi, z, intr))
        P_obj_list.append(P_obj)

    P_cam_arr = np.array(P_cam_list, dtype=np.float64)
    P_obj_arr = np.array(P_obj_list, dtype=np.float64)
    if P_cam_arr.shape[0] < MIN_VALID_CORNERS:
        return False, None, None, "Not enough valid depth corners", all_corners, ids

    # A) Hybrid: rotation from PnP, translation from depth corners
    ok_pnp, rvec, _tvec = cv2.solvePnP(
        obj_pts_marker.astype(np.float32),
        corners_uv.astype(np.float32),
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if ok_pnp:
        R_cam_from_marker, _ = cv2.Rodrigues(rvec)
        t_cam_from_marker = estimate_translation_from_depth_corners(
            R_cam_from_marker, corners_uv, obj_pts_marker, depth_frame, intr, NEIGHBOR_RADIUS_PX
        )
        if t_cam_from_marker is not None:
            R = R_cam_from_marker
            t = t_cam_from_marker
            method = "Hybrid (R from PnP, t from depth)"
        else:
            R, t, method = None, None, ""
    else:
        R, t, method = None, None, ""

    # B) Fallback: full pose from 3D-3D (Kabsch)
    if R is None:
        R, t = kabsch_rigid_transform(P_obj_arr, P_cam_arr)
        if R is None:
            return False, None, None, "Pose failed", all_corners, ids
        method = "Kabsch (pose from depth corners)"

    # marker->camera: P_cam = R*P_marker + t
    # camera->marker:
    R_marker_from_cam = R.T
    t_marker_from_cam = -R_marker_from_cam @ t
    return True, R_marker_from_cam, t_marker_from_cam, method, all_corners, ids


def detect_ball_center(frame_bgr, bs, last_pts):
    """Returns (found, (u,v), debug_info) using your white+motion style filter."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, V_LOW), (179, S_HIGH, 255))
    k5 = np.ones((5, 5), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k5, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k5, iterations=2)

    fg = bs.apply(frame_bgr, learningRate=0.002)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
    fg = cv2.dilate(fg, k5, iterations=1)

    mask = cv2.bitwise_and(white, fg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pred = None
    if len(last_pts) >= 2:
        (x1, y1), (x2, y2) = last_pts[-2], last_pts[-1]
        pred = (x2 + (x2 - x1), y2 + (y2 - y1))
    elif len(last_pts) == 1:
        pred = last_pts[-1]

    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < AREA_MIN or area > AREA_MAX:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solid = (area / hull_area) if hull_area > 0 else 0.0

        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w / max(1, h), h / max(1, w))

        if circ < CIRC_MIN or solid < SOLID_MIN or aspect > ASPECT_MAX:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dist_pred = 0.0
        if pred is not None:
            dist_pred = math.hypot(cx - pred[0], cy - pred[1])

        score = (-2.0 * dist_pred) + (0.25 * area) + (350.0 * circ) + (250.0 * solid) - (60.0 * aspect)

        if best is None or score > best["score"]:
            best = dict(score=score, cx=cx, cy=cy, hull=hull, bbox=(x, y, w, h))

    if best is None:
        return False, None, dict(mask=mask)

    u, v = int(round(best["cx"])), int(round(best["cy"]))
    last_pts.append((best["cx"], best["cy"]))
    if len(last_pts) > 10:
        last_pts[:] = last_pts[-10:]
    return True, (u, v, best), dict(mask=mask)


# ---------------- MAIN ----------------
def main():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    obj_pts_marker = create_marker_object_points(MARKER_LENGTH_M)

    bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    print("Warming up background model...")
    for _ in range(WARMUP_FRAMES):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color = frames.get_color_frame()
        if not color:
            continue
        frame = np.asanyarray(color.get_data())
        bs.apply(frame, learningRate=0.05)
    print("Running. Press 'q' to quit.")

    last_pts = []

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

            # 1) camera -> marker transform (R,t)
            ok_pose, R_m_c, t_m_c, method, all_corners, ids = get_camera_to_marker_transform(
                frame, depth, intr, K, dist, dictionary, params, obj_pts_marker
            )

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, all_corners, ids)

            if ok_pose:
                # Draw axes (need marker->camera pose; invert back for drawing)
                # marker->camera: R_c_m = R_m_c^T, t_c_m = -R_c_m * t_m_c
                R_c_m = R_m_c.T
                t_c_m = -R_c_m @ t_m_c
                rvec_draw, _ = cv2.Rodrigues(R_c_m)
                tvec_draw = t_c_m.reshape(3, 1)
                cv2.drawFrameAxes(vis, K, dist, rvec_draw, tvec_draw, MARKER_LENGTH_M * 0.75)

            # 2) ball detection
            found_ball, ball_info, dbg = detect_ball_center(frame, bs, last_pts)
            if found_ball:
                u, v, best = ball_info
                cv2.circle(vis, (u, v), 5, (255, 0, 0), -1)
                cv2.drawContours(vis, [best["hull"]], -1, (0, 255, 0), 2)

                # depth -> 3D in camera frame
                z = robust_depth_at_pixel(depth, u, v, BALL_DEPTH_RADIUS_PX)
                if z > 0:
                    P_ball_cam = deproject(u, v, z, intr)  # meters

                    if ok_pose:
                        # Transform to marker frame: P_marker = R_m_c * P_cam + t_m_c
                        P_ball_marker = (R_m_c @ P_ball_cam) + t_m_c
                        xM, yM, zM = P_ball_marker.tolist()

                        draw_overlay(
                            vis,
                            [
                                method,
                                f"Ball pixel: u={u}, v={v}",
                                f"Ball depth (m): {z:.3f}",
                                f"Ball XYZ in marker67 (m): x={xM:.3f}, y={yM:.3f}, z={zM:.3f}",
                            ],
                            x=10, y=25, font_scale=0.65, color=(0, 255, 0)
                        )
                    else:
                        draw_overlay(
                            vis,
                            [
                                "Marker pose not available",
                                f"Ball pixel: u={u}, v={v}",
                                f"Ball depth (m): {z:.3f}",
                            ],
                            x=10, y=25, font_scale=0.65, color=(0, 0, 255)
                        )
                else:
                    draw_overlay(
                        vis,
                        ["Ball found but depth invalid at center"],
                        x=10, y=25, font_scale=0.65, color=(0, 0, 255)
                    )
            else:
                draw_overlay(
                    vis,
                    [("Pose: " + method) if ok_pose else "Pose: marker not found",
                     "Ball: not found"],
                    x=10, y=25, font_scale=0.65, color=(255, 255, 0)
                )

            cv2.imshow("ball_xyz_marker67", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()