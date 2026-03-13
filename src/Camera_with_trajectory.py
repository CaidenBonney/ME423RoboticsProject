import os
import time
from typing import Optional
import numpy as np
import cv2
import pyrealsense2 as rs
import math
from Trajectory import Trajectory, update_trajectory

class Camera:
    def __init__(self) -> None:
        self.startTime = time.time()
        self.sampleRate = 30
        self.sampleTime = 1 / self.sampleRate
        self.bs = None  # background susbtractor mpdel
        self.intrinsics = None  # camera intrinsics
        self.robotTransformation = None  # transformation from camera frame to robot frame
        self.pipeline = None  # realsense pipeline
        self.profile = None  # realsense pipeline profile
        self.align = None  # realsense align object
        self.cameraPortID = 3  # This number may be different for every machine. It corresponds to the port that the camera is attached to
        self.camera_matrix = None
        self.dist_coeffs = None
        self.R_m_c = None
        self.t_m_c = None
        self.current_frame = np.zeros([640, 480, 3], dtype=np.uint8)
        self.trajectory = Trajectory(0, 0, 0, 0)
        self.cam_setup()

    def elapsed_time(self) -> float:
        return time.time() - self.startTime

    def capture_image(self) -> rs.align:
        """ Captures an RGB and depth frame from the camera. 
        Images are aligned."""
        frames = self.pipeline.wait_for_frames()
        return self.align.process(frames)

    def cam_setup(self) -> None:
        # TODO: add camera initialization (device open, stream config, etc.).
        # configure depth and color streamss
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.profile = self.pipeline.start(cfg)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # cv2.namedWindow('depth_cam', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('rgb_cam', cv2.WINDOW_AUTOSIZE)
        self.startTime = time.time()
        self.cam_calibration()

    def cam_calibration(self, path: str = "camera_calib.yml") -> None:
        self.camera_matrix = np.array(([800, 0, 320], [0, 800, 240], [0, 0, 1]), dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        print("SOLVING ROBOT TRANSFORMATION ...")
        self.get_robot_transformation()
        print("ROBOT TRANSFORMATION SOLVED...")
        self.create_background_model()
        print("BACKGROUND MODEL CREATED...")

    def get_robot_transformation(self) -> np.ndarray:
        MARKER_ID = 67
        MARKER_LENGTH_M = 0.07
        ARUCO_DICT = cv2.aruco.DICT_4X4_250

        W, H, FPS = 640, 480, 30

        NEIGHBOR_RADIUS_PX = 2  # depth sampling neighborhood for marker corners
        BALL_DEPTH_RADIUS_PX = 2  # depth sampling neighborhood for ball center
        MIN_VALID_CORNERS = 3

        marker_found = False
        while not marker_found:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color = aligned_frames.get_color_frame()
            depth = aligned_frames.get_depth_frame()
            frame = np.asanyarray(color.get_data())
            vis = frame.copy()
            K = np.array(
                [[self.intrinsics.fx, 0, self.intrinsics.ppx], [0, self.intrinsics.fy, self.intrinsics.ppy], [0, 0, 1]],
                dtype=np.float64,
            )
            dist = np.zeros((5, 1), dtype=np.float64)

            dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            params = cv2.aruco.DetectorParameters()
            obj_pts_marker = create_marker_object_points(MARKER_LENGTH_M)

            # 1) camera -> marker transform (R,t)
            ok_pose, R_m_c, t_m_c, method, all_corners, ids = get_camera_to_marker_transform(
                frame, depth, self.intrinsics, K, dist, dictionary, params, obj_pts_marker
            )

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, all_corners, ids)
                self.current_frame = vis
                print("ARUCO MARKER DETCTED ...")
                cv2.waitKey(0)

            if ok_pose:
                print("Camera to marker pose found")
                # Draw axes (need marker->camera pose; invert back for drawing)
                # marker->camera: R_c_m = R_m_c^T, t_c_m = -R_c_m * t_m_c
                R_c_m = R_m_c.T
                t_c_m = -R_c_m @ t_m_c
                rvec_draw, _ = cv2.Rodrigues(R_c_m)
                tvec_draw = t_c_m.reshape(3, 1)
                cv2.drawFrameAxes(vis, K, dist, rvec_draw, tvec_draw, MARKER_LENGTH_M * 0.75)
                self.current_frame = vis
                self.R_m_c = R_m_c
                self.t_m_c = t_m_c
                self.robotTransformation = build_T(R_m_c, t_m_c)
                marker_found = True
            else:
                print("Camera to marker pose NOT found, using identity")
                self.R_m_c = np.eye(3, dtype=np.float64)
                self.t_m_c = np.zeros((3,), dtype=np.float64)
                self.robotTransformation = np.eye(4, dtype=np.float64)

    def create_background_model(
        self,
        warm_up_video_path: str = "src/videos/warmup_video.mp4",
        warmup_frames: int = 30,
        fps: int = 60,
        W: int = 640,
        H: int = 480,
    ) -> None:
        # TODO: add camera calibration logic (make rerunable).
        # update transformation from camera frame to robot frame, create background model for motion detection
        warmup_frames_count = 0
        if os.path.exists(warm_up_video_path):
            try:
                os.remove(warm_up_video_path)
                print(f"Deleted old video: {warm_up_video_path}")
            except Exception as e:
                print(f"Could not delete video {warm_up_video_path}: {e}")
        else:
            print(f"No existing video to delete at: {warm_up_video_path}")
        # initialize background subtractor
        cap = cv2.VideoCapture(
            self.cameraPortID
        )  # This number may be different for every machine. It corresponds to the port that the camera is attached to
        writer = cv2.VideoWriter(
            warm_up_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
            True,
        )
        # record warmup frames to video
        while warmup_frames_count < warmup_frames:
            ret, frame = cap.read()
            if not ret:
                print("failed to grab frame")
                break
            else:
                writer.write(frame)
                warmup_frames_count += 1
        writer.release()
        cap.release()
        # Background subtractor to remove static bright objects (like the screw)
        self.bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        # Warm up background model
        for i in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.bs.apply(frame, learningRate=0.05)
        print("WARMED UP BACKGROUND MODEL...")
        cap.release()

    pass

    def image_processing(self, aligned_frames: rs.align) -> np.ndarray:
        # TODO: convert image to XYZ
        rgb = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        depth_image = np.asarray(depth.get_data(), dtype=np.uint8)
        frame = rgb.get_data()
        frame = np.asanyarray(frame)
        self.current_frame = frame # store frame so trajectory can be drawn on and display
        vis = frame.copy()
        last_pts = []
        # 2) ball detection
        found_ball, ball_info, mask = detect_ball_center(frame, self.bs, last_pts)
        if found_ball:
            u, v, best = ball_info
            cv2.circle(vis, (u, v), 5, (255, 0, 0), -1)
            cv2.drawContours(vis, [best["hull"]], -1, (0, 255, 0), 2)
            # depth -> 3D in camera frame
            z = robust_depth_at_pixel(depth, u, v, BALL_DEPTH_RADIUS_PX)
            if z > 0:
                P_ball_cam = deproject(u, v, z, self.intrinsics)  # meters
                # Transform to marker frame: P_marker = R_m_c * P_cam + t_m_c
                P_ball_marker = (self.R_m_c @ P_ball_cam) + self.t_m_c
                xM, yM, zM = P_ball_marker.tolist()
                return np.asarray([xM, yM, zM], dtype=np.float64)
        return np.asarray([-1, -1, -1], dtype=np.float64)

    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self) -> Optional[np.ndarray]:
        """ Captures an RGB and depth frame from the camera and outputs the ball XYZ (w.r.t base coords)"""
        # Grab the latest RGBD frames
        RBGD_frames = self.capture_image()
        if RBGD_frames is None:
            return None

        
        # Obtain Ball XYZ from RGBD frame in 
        XYZ = self.image_processing(RBGD_frames)
    
        # change coordinates into camera frame so trajectory can be projected onto frame
        XYZ_cam = np.linalg.solve(self.R_m_c, np.asanyarray(XYZ) - self.t_m_c)

        # XYZ = self.image_processing(input)

        # XYZ = np.random.uniform(
        #     low=np.array([0.50, -0.10, 0.55], dtype=np.float64),
        #     high=np.array([0.40, 0.10, 0.45], dtype=np.float64),
        # )

        color = (0, 0, 255)   # Red color in BGR
        marker_type = cv2.MARKER_STAR
        marker_size = 30
        thickness = 2
        sliding_window_size = 5
        line_type = cv2.LINE_AA # Anti-aliased line for smoother appearance
        t = float(time.time() - self.startTime)
        # Draw the marker
        self.trajectory = update_trajectory(t, XYZ_cam, sliding_window_size)
        for i in range(20):
            t += 0.05 # timestep in seconds
            # print("self.trajectory.pos(t): ", self.trajectory.pos(t).reshape(3,))
            pos_list = [self.trajectory.pos(t).reshape(3,)[i] for i in range(3)]
            print("self.intrinsics type: ", type(self.intrinsics))
            position = tuple(rs.rs2_project_point_to_pixel(self.intrinsics, pos_list)) # project trajectory point back to pixel coordinates for drawing. Need intrinsics for this, so may need to check type requirement for XYZ_cam
            # need to check type requirement for XYZ_cam
            print("projected position: ", position)
            position = (int(position[0]), int(position[1]))
            cv2.drawMarker(self.current_frame, position, color, markerType=marker_type, markerSize=marker_size, thickness=thickness, line_type=line_type)
        print("ball position in Robot Coordinates: ", XYZ)
        return XYZ


# ---------------- USER CONFIG ----------------
MARKER_ID = 67
MARKER_LENGTH_M = 0.07
ARUCO_DICT = cv2.aruco.DICT_4X4_250

W, H, FPS = 640, 480, 30

NEIGHBOR_RADIUS_PX = 2  # depth sampling neighborhood for marker corners
BALL_DEPTH_RADIUS_PX = 2  # depth sampling neighborhood for ball center
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


# ---------------- MARKER MODEL ----------------
def create_marker_object_points(marker_length_m: float) -> np.ndarray:
    L = marker_length_m
    return np.array(
        [
            [-L / 2, L / 2, 0],
            [L / 2, L / 2, 0],
            [L / 2, -L / 2, 0],
            [-L / 2, -L / 2, 0],
        ],
        dtype=np.float64,
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
    T[:3, 3] = t.reshape(
        3,
    )
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


def estimate_translation_from_depth_corners(
    R_cam_from_marker: np.ndarray,
    corners_uv: np.ndarray,
    obj_pts_marker: np.ndarray,
    depth_frame,
    intr: rs.intrinsics,
    radius_px: int,
) -> np.ndarray | None:
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


def get_camera_to_marker_transform(frame_bgr, depth_frame, intr, K, dist, dictionary, params, obj_pts_marker):
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
        obj_pts_marker.astype(np.float32), corners_uv.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
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
