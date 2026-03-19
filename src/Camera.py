import gc
import os
import time
from typing import Optional
import numpy as np
import cv2
import pyrealsense2 as rs
import math

# ---------------- USER CONFIG ----------------
MARKER_ID = 67 # ArUco marker ID
MARKER_LENGTH_M = 0.1889  # ArUco marker side length in meters (0.1889 m = 7.437 inches)
ARUCO_DICT = cv2.aruco.DICT_4X4_250 # ArUco dictionary

W, H, FPS = 640, 480, 60 # Camera resolution and FPS

NEIGHBOR_RADIUS_PX = 2  # depth sampling neighborhood for marker corners
BALL_DEPTH_RADIUS_PX = 4  # depth sampling neighborhood for ball center
MIN_VALID_CORNERS = 3 # minimum number of corners to consider a valid marker

# Ball detector modes
WHITE_BALL_COLOR = 0
ORANGE_BALL_COLOR = 1
GREEN_BALL_COLOR = 2
SPIKE_BALL_COLOR = 3


# White mask values
S_HIGH = 70
V_LOW = 185

# Ball detector thresholds (tune if needed)
AREA_MIN = 4
AREA_MAX = 12000
CIRC_MIN = 0.25
SOLID_MIN = 0.40
ASPECT_MAX = 1.8

WARMUP_FRAMES = 30 # number of frames to capture is feed to background subtractor

CALIBRATION_FILE = "camera_calib.yml"


class Camera:
    """Camera class for handling RealSense D435 camera, ball detection, and position tracking."""
    def __init__(self) -> None:
        self.startTime = time.time() # time since camera was initialized
        self.sampleRate = FPS  # [Hz] how often to capture/process frames from the camera. Can be lower than the camera FPS to reduce noise and CPU load.
        self.sampleTime = 1 / self.sampleRate
        
        self.color_intrinsics = None # RGB camera intrinsics
        self.depth_intrinsics = None # depth camera intrinsics
        self.aligned_depth_intrinsics = None # aligned depth camera intrinsics

        self.pipeline = None  # realsense pipeline
        self.profile = None  # realsense pipeline profile
        self.align = None  # realsense align object
        self.cameraPortID = 3  # This number may be different for every machine. It corresponds to the port that the camera is attached to

        
        # Camera calibration (focal length, principal point, distortion coefficients)
        self.camera_matrix = None # No longer used, experimentally gathered
        self.dist_coeffs = None # No longer used, experimentally gathered

        self.bs = None  # background susbtractor model

        self.u: Optional[int] = None # "x" coordinate of ball in aligned camera frame [pixels]
        self.v: Optional[int] = None # "y" coordinate of ball in aligned camera frame [pixels]
        self.z: Optional[float] = None # depth measurement of ball from camera [meters] (This is a distance normal from the camera to the ball)

        # temp variables for ball detection, logged best detection score, used for debugging
        self.score = None
        self.score_parts = None

        # Relevant Transformations
        self.R_m_c = None # rotation matrix from camera frame to ArUco frame [3x3]
        self.t_m_c = None # translation vector from camera frame to ArUco frame [3x3]
        self.T_cam2ArUco = None  # transformation from camera frame to ArUco frame [4x4]
        self.T_cam2ArUco_inv = None  # inverse of T_cam2ArUco

        ## transformation from ArUco frame to base frame [4x4] (measured and tuned)
        self.ArUco2Base_Transformation = np.array(
            [[1, 0, 0, 0.622], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        ) 

        ## inverse of ArUco2Base_Transformation
        self.ArUco2Base_Transformation_inv = np.linalg.inv(self.ArUco2Base_Transformation)

        # RGB frame, used for upstream display and debugging
        self.current_frame = np.zeros((640, 480, 3), dtype=np.uint8)

        # Runs camera setup processes
        self.cam_setup()

    def elapsed_time(self) -> float:
        """Time elapsed since camera was initialized."""
        return time.time() - self.startTime

    def capture_image(self) -> rs.align:
        """Captures an RGB and depth frame from the camera.
        Images are aligned."""

        # Grab frames from hardware (aligned)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        rgb = aligned_frames.get_color_frame()

        # depth = aligned_frames.get_depth_frame()
        # depth_image = np.asarray(depth.get_data(), dtype=np.uint8)
        
        frame = rgb.get_data()
        frame = np.asanyarray(frame)
        self.current_frame = np.asanyarray(frame, dtype=np.uint8)
        return aligned_frames

    def cam_setup(self) -> None:
        """Configures RealSense D435 camera, starts streaming, and configures camera alignment using ArUco (blocking)."""
        # configure depth and color streamss
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(cfg)

        # align to color stream
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # because depth is aligned to color in this pipeline, use color intrinsics for aligned depth pixels
        self.aligned_depth_intrinsics = self.color_intrinsics

        self.startTime = time.time()
        self.cam_calibration()

    def cam_calibration(self, path: str = "camera_calib.yml") -> None:
        """Executes the following camera setup steps:
        1. Loads the camera calibration file to remove fish eye distortion.
        2. Solves the camera to robot transformation.
        3. Creates the background subtractor model.
        """

        # How to load calibration information: not used anymore
        self.camera_matrix, self.dist_coeffs = load_calibration(CALIBRATION_FILE)

        print("SOLVING CAMERA TO ROBOT TRANSFORMATION ...")
        self.get_robot_transformation()
        print("ROBOT TRANSFORMATION SOLVED...")
        self.create_background_model()
        print("BACKGROUND MODEL CREATED...")

    def get_robot_transformation(self) -> None:
        """ Finds the camera to robot transformation using ArUco markers. (BLOCKING)"""
        
        marker_found = False
        while not marker_found:
            # Grab frames from hardware (aligned)
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color = aligned_frames.get_color_frame()
            depth = aligned_frames.get_depth_frame()
            frame = np.asanyarray(color.get_data())

            # camera intrinsics from RealSense SDK
            K = np.array(
                [[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx], [0, self.color_intrinsics.fy, self.color_intrinsics.ppy], [0, 0, 1]],
                dtype=np.float64,
            )
            dist = np.zeros((5, 1), dtype=np.float64)

            # ArUco marker detection
            dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            params = cv2.aruco.DetectorParameters()
            obj_pts_marker = create_marker_object_points(MARKER_LENGTH_M)

            # 1) camera -> marker transform (R,t)
            ok_pose, R_m_c, t_m_c, method, all_corners, ids = get_camera_to_marker_transform(
                frame, depth, self.aligned_depth_intrinsics, K, dist, dictionary, params, obj_pts_marker
            )

            if ids is not None:
                print("AN ARUCO MARKER WAS DETECTED ...")

            if ok_pose:
                print("Camera to marker pose found")
                self.R_m_c = R_m_c # rotation matrix from camera frame to ArUco frame [3x3]
                self.t_m_c = t_m_c # translation vector from camera frame to ArUco frame [3x3]
                self.T_cam2ArUco = build_T(self.R_m_c, self.t_m_c)
                self.T_cam2ArUco_inv = np.linalg.inv(self.T_cam2ArUco)
                marker_found = True
            else:
                print("Camera to marker pose NOT found, using identity")
                self.R_m_c = np.eye(3, dtype=np.float64)
                self.t_m_c = np.zeros((3,), dtype=np.float64)
                self.T_cam2ArUco = np.eye(4, dtype=np.float64)

    def create_background_model(
        self,
        warm_up_video_path: str = "src/videos/warmup_video.mp4",
        warmup_frames: int = 45,
        fps: int = FPS,
        W: int = 640,
        H: int = 480,
    ) -> None:

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
        # # initialize background subtractor
        # cap = cv2.VideoCapture(
        #     self.cameraPortID
        # )  # This number may be different for every machine. It corresponds to the port that the camera is attached to
        writer = cv2.VideoWriter(
            warm_up_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
            True,
        )
        # # record warmup frames to video
        # while warmup_frames_count < warmup_frames:
        #     ret, frame = cap.read()
        #     if not ret:
        #         print("failed to grab frame")
        #         break
        #     else:
        #         writer.write(frame)
        while warmup_frames_count < warmup_frames:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color = aligned_frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            writer.write(frame)
            warmup_frames_count += 1
        writer.release()
        # cap.release()
        # Background subtractor to remove static bright objects (like the screw)
        self.bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        # Warm up background model
        cap = cv2.VideoCapture(warm_up_video_path)
        for i in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.bs.apply(frame, learningRate=0.05)
        print("WARMED UP BACKGROUND MODEL...")
        cap.release()
        gc.collect()

    pass

    def image_processing(self, aligned_frames: rs.align) -> tuple[np.ndarray, bool]:
        """Process the aligned RGBD frames and return the ball position in camera coordinates.
        Args:
            aligned_frames (rs.align): The aligned RGBD frames.

        Returns:
            np.ndarray: The ball position in camera coordinates or None if no ball is detected.
            bool: whether or not the ball has been found in the frame
        """
        timestamp = aligned_frames.get_timestamp()
        rgb = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        depth_image = np.asarray(depth.get_data(), dtype=np.uint8)
        frame = rgb.get_data()
        frame = np.asanyarray(frame)
        vis = frame.copy()

        # ball detection
        found_ball, ball_info, mask = detect_ball_center(frame, self.bs, ball_color=GREEN_BALL_COLOR)
        if found_ball:
            self.u, self.v, best = ball_info
            self.score = best["score"]
            self.score_parts = best["score_parts"]
            cv2.circle(self.current_frame, (self.u, self.v), 5, (255, 0, 0), -1)
            cv2.drawContours(self.current_frame, [best["hull"]], -1, (0, 255, 0), 2)

            # depth -> 3D in camera frame
            self.z = robust_depth_at_pixel(depth, self.u, self.v, BALL_DEPTH_RADIUS_PX)
            if self.z > 0:
                P_ball_cam = deproject(self.u, self.v, self.z, self.aligned_depth_intrinsics)  # meters
                # Transform from camera frame to robot base frame
                xR, yR, zR = self.T_Camera_to_RobotBase(P_ball_cam)

                return (np.asarray([xR, yR, zR], dtype=np.float64), found_ball)
            else:
                # print("Invalid depth for ball ...")
                pass
        else:
            # print("BALL NOT DETECTED ...")
            pass
        return (None, False)

    def T_Camera_to_RobotBase(self, P_ball_cam: np.ndarray) -> tuple[float, float, float]:
        """Transform from camera frame to robot frame

        Args:
            P_ball_cam (np.ndarray): 3x1 vector in camera frame (output of deproject)

            Returns:
                xR, yR, zR (float): In robot base frame
        """
        #                                   turn 3x1 vector into 4x1 vector
        P_ball_ArUco = self.T_cam2ArUco @ np.append(P_ball_cam, 1.0).reshape(4, 1)
        P_ball_base = self.ArUco2Base_Transformation @ P_ball_ArUco
        xR, yR, zR = P_ball_base[:3, 0]
        return xR, yR, zR

    def T_RobotBase_to_Camera(self, XYZR: np.ndarray) -> tuple[int, int]:
        """Transform from Robot base frame to camera frame

        Args:
            XYZR (np.ndarray): 3x1 vector in robot base frame

        Returns:
            u, v, [int]"""
        # turn 3x1 vector into 4x1 vector
        P_ball_base = np.append(XYZR, 1.0).reshape(4, 1)

        # Convert from robot base frame to ArUco frame
        if self.ArUco2Base_Transformation_inv is None:
            raise ValueError("ArUco2Base_Transformation_inv is None")
        P_ball_ArUco = self.ArUco2Base_Transformation_inv @ P_ball_base

        # Convert from ArUco frame to camera frame
        if self.T_cam2ArUco_inv is None:
            raise ValueError("T_cam2ArUco_inv is None")
        P_ball_cam = self.T_cam2ArUco_inv @ P_ball_ArUco
        # Project XYZ_cam to pixel coordinates
        u, v = tuple(
            rs.rs2_project_point_to_pixel(
                self.color_intrinsics,
                [
                    P_ball_cam[:3, 0].reshape(
                        3,
                    )[i]
                    for i in range(3)
                ],
            )
        )
        return (int(u), int(v))

    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self) -> tuple[Optional[np.ndarray], bool, Optional[float]]:
        """Captures an RGB and depth frame from the camera and outputs the ball XYZ (w.r.t base coords)"""
        # Grab the latest RGBD frames
        RBGD_frames = self.capture_image()
        if RBGD_frames is None:
            print("Failed to capture image")
            return None, False, None
        timestamp = RBGD_frames.get_timestamp()

        # Obtain Ball XYZ from RGBD frame
        XYZ, ball_found = self.image_processing(RBGD_frames)

        return XYZ, ball_found, timestamp

    def show_image(self):
        cv2.waitKey(1)  # wait 1 ms. needed to display video feed
        cv2.imshow("camera_pov", self.current_frame)


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
    """Estimates the depth at a pixel location using a radius-based median filter."""
    
    # If radius is <= 0, return the depth at the specified pixel location
    if radius <= 0:
        return float(depth_frame.get_distance(u, v))
    vals = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu, vv = u + du, v + dv

            # Try to get the depth at the specified pixel location, errors from pixels out of bounds are caught and ignored
            try:
                z = float(depth_frame.get_distance(uu, vv))
                if z > 0:
                    vals.append(z)
            except:
                pass
    if not vals:
        return 0.0
    return float(np.median(vals))


def deproject(u: int, v: int, depth_m: float, intr: rs.intrinsics) -> np.ndarray:
    """Calls RealSense SDK to deproject a pixel location to 3D coordinates in the camera frame. 
    
    Args:
        u (int): The "x" coordinate of the pixel location in the aligned camera frame.
        v (int): The "y" coordinate of the pixel location in the aligned camera frame.
        depth_m (float): The depth of the pixel location in the aligned camera frame.
        intr (rs.intrinsics): The depth camera intrinsics.

    Returns:
        p (np.ndarray): The 3D coordinates of the pixel location in the camera frame.
    """

    p = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
    return np.array(p, dtype=np.float64)  # [X,Y,Z] meters in camera frame


def build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Builds a 4x4 homogeneous transformation matrix from rotation and translation vectors."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def kabsch_rigid_transform(A: np.ndarray, B: np.ndarray):
    """Uses Kabsch algorithm to find the optimal rotation and translation to align two sets of points."""
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
    """Uses CV2 to detect ArUco markers and their corners. Reformats data for easier processing.
    
    Args:
        frame_bgr (np.ndarray): The frame to detect the marker in.
        marker_id (int): The ID of the marker to detect.
        dictionary (cv2.aruco.getPredefinedDictionary): The marker dictionary.
        params (cv2.aruco.DetectorParameters): The marker detector parameters.

    Returns:
        found (bool): True if the specified marker was detected, False otherwise.
        corners_uv (np.ndarray): The corners of the detected marker in the camera frame.
        all_corners (np.ndarray): The corners of all detected markers in the camera frame.
        ids (np.ndarray): The IDs of all detected markers.
    """
    
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
    """Estimates the translation vector from the corners of a marker in the camera frame to the object frame.
    
    Args:
        R_cam_from_marker (np.ndarray): The rotation matrix from the camera frame to the marker frame.
        corners_uv (np.ndarray): The corners of the marker in the camera frame.
        obj_pts_marker (np.ndarray): The corners of the marker in the object frame.
        depth_frame (rs.frame): The depth frame to estimate the camera to marker transform in.
        intr (rs.intrinsics): The camera intrinsics.
        radius_px (int): The radius of the median filter to use for robust depth estimation.

    Returns:
        t_list (np.ndarray): The translation vector from the corners of the marker in the camera frame to the object frame.
    """
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
    Uses the marker corners to estimate the camera to marker transform.

    Args:
        frame_bgr (np.ndarray): The RGB frame to estimate the camera to marker transform in.
        depth_frame (rs.frame): The depth frame to estimate the camera to marker transform in.
        intr (rs.intrinsics): The camera intrinsics.
        K (np.ndarray): The camera matrix.
        dist (np.ndarray): The camera distortion coefficients.
        dictionary (cv2.aruco.getPredefinedDictionary): The marker dictionary.
        params (cv2.aruco.DetectorParameters): The marker detector parameters.
        obj_pts_marker (np.ndarray): The marker corners in object coordinates.

    Returns:
        (ok: Boolean,
        Optional[np.ndarray]: Rotation matrix from camera to marker,
        Optional[np.ndarray]: Translation vector from camera to marker,
        Optional[str]: method used,
        Optional[List[np.ndarray]]: all detected corners,
        Optional[np.ndarray]: all detected ids)
    """
    found, corners_uv, all_corners, ids = detect_marker_corners(frame_bgr, MARKER_ID, dictionary, params)
    if not found:
        return False, None, None, "Marker not found", all_corners, ids

    # gather 3D corner points from depth
    P_cam_list = [] # list of 3D points in camera frame
    P_obj_list = [] # list of 3D points in object frame

    # For each corner in the marker, estimate the depth at the corner and deproject it to the camera frame
    for (u, v), P_obj in zip(corners_uv, obj_pts_marker):
        ui, vi = int(round(u)), int(round(v))
        z = robust_depth_at_pixel(depth_frame, ui, vi, NEIGHBOR_RADIUS_PX)
        if z <= 0:
            continue
        P_cam_list.append(deproject(ui, vi, z, intr))
        P_obj_list.append(P_obj)

    # Convert lists to arrays
    P_cam_arr = np.array(P_cam_list, dtype=np.float64)
    P_obj_arr = np.array(P_obj_list, dtype=np.float64)
    if P_cam_arr.shape[0] < MIN_VALID_CORNERS:
        return False, None, None, "Not enough valid depth corners", all_corners, ids

    # Issue to fix: translation vector from camera frame to marker was inconsistent
    # Option A) Hybrid: rotation from PnP, translation from depth corners
    ok_pnp, rvec, _tvec = cv2.solvePnP(
        obj_pts_marker.astype(np.float32), corners_uv.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if ok_pnp:
        R_cam_from_marker, _ = cv2.Rodrigues(rvec)
        # Need to estimate the translation vector from the corners of the marker in the camera frame to the object frame
        t_cam_from_marker = estimate_translation_from_depth_corners(
            R_cam_from_marker, corners_uv, obj_pts_marker, depth_frame, intr, NEIGHBOR_RADIUS_PX
        )
        if t_cam_from_marker is not None:
            R = R_cam_from_marker # R is the rotation matrix to the camera frame frome the marker frame
            t = t_cam_from_marker # t is the translation vector to the camera frame from the marker frame
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

    R_marker_from_cam = R.T
    t_marker_from_cam = -R_marker_from_cam @ t
    return True, R_marker_from_cam, t_marker_from_cam, method, all_corners, ids


def detect_ball_center(frame_bgr, bs, ball_color: int = WHITE_BALL_COLOR, using_bg_sub: bool = True):
    """Detects the ball center in the frame using the frame, background subtractor and last detected points.

    Args:
        frame_bgr (np.ndarray): The frame to detect the ball center in.
        bs (cv2.BackgroundSubtractorMOG2): The background subtractor to use.
        last_pts (list): The last detected points.
        ball_color (int): The color of the ball to detect. Use the constants WHITE_BALL_COLOR, ORANGE_BALL_COLOR, or GREEN_BALL_COLOR.

    Returns:
      (found: Boolean, Optional[(u,v, best: dict)], "dict(mask=mask)")"""
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    k5 = np.ones((5, 5), np.uint8)

    # Depending on the ball color selected, create color mask 
    if ball_color == WHITE_BALL_COLOR:
        color_mask = cv2.inRange(hsv, (0, 0, V_LOW), (179, S_HIGH, 255))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    elif ball_color == ORANGE_BALL_COLOR:
        lower_orange = (7, 160, 130)
        upper_orange = (18, 255, 255)

        color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    elif ball_color == GREEN_BALL_COLOR:
        lower_green = (70, 50, 40)
        upper_green = (90, 255, 255)
        using_bg_sub = True  # bg sub seems to hurt green ball detection, so disable for green ball

        color_mask = cv2.inRange(hsv, lower_green, upper_green)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    elif ball_color == SPIKE_BALL_COLOR:
        # ** THESE VALUES ARE MADE UP, need to tune**
        lower_yellow = (70, 50, 40)
        upper_yellow = (90, 255, 255)

        color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    else:
        raise ValueError(f"Invalid ball color: {ball_color}")

    # If background subtraction is used, apply it to the frame
    if using_bg_sub:
        k3 = np.ones((3, 3), np.uint8)

        fg = bs.apply(frame_bgr, learningRate=0.002)
        # erosion and dilation to remove small balls
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
        # dilate fill gaps in ball contour, use larger kernel for better results
        fg = cv2.dilate(fg, k5, iterations=1)
        # Only pixels captured by color mask AND bg subtraction are considered
        mask = cv2.bitwise_and(fg, color_mask)
    else:
        mask = color_mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    # For each contour, calculate the circularity and solidty of the contour
    for c in contours:
        area = cv2.contourArea(c)
        # Area filter
        if area < AREA_MIN or area > AREA_MAX:
            continue

        # Must have perimeter
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue

        # Calculate circularity
        circ = 4 * math.pi * area / (peri * peri)

        # Calculate solidty
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solid = (area / hull_area) if hull_area > 0 else 0.0

        # mean_hue, _ = mean_hue_in_hull(frame_bgr, hull)
        # hue_diff = abs(mean_hue - mean_green[0])

        # bounding box dims
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w / max(1, h), h / max(1, w))

        # circularity, solidity, and bounding box aspect ratio filtering
        if circ < CIRC_MIN or solid < SOLID_MIN or aspect > ASPECT_MAX:
            continue

        # find centroid
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # score based off calculated metrics
        dist_score = 0 # no longer used
        area_score = 0.25 * area # no longer used: only logged
        circ_score = 350.0 * circ # circularity score
        solid_score = 250.0 * solid # solidity score
        aspect_score = -60.0 * aspect # bounding box aspect ratio score
        # color_score = -90.0 * hue_diff
        color_score = 0.0  # no longer used (used to do above)

        # best score for this contour
        score = dist_score + circ_score + solid_score + aspect_score + color_score

        if best is None or (score > best["score"]):
            best = dict(
                score=score,
                cx=cx,
                cy=cy,
                hull=hull,
                bbox=(x, y, w, h),
                score_parts=(dist_score, area_score, circ_score, solid_score, aspect_score, color_score),
            )

    if best is None:
        return False, None, dict(mask=mask)

    u, v = int(round(best["cx"])), int(round(best["cy"]))
    return True, (u, v, best), dict(mask=mask)


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


def mean_hue_in_hull(frame_bgr, hull):
    """
    Compute the circular mean hue inside an OpenCV convex hull.

    Args:
        frame_bgr: HxWx3 uint8 BGR image
        hull: OpenCV convex hull, shape Nx1x2 or Nx2

    Returns:
        mean_hue_opencv: float in [0, 180)
        valid_pixel_count: int
    """
    import cv2
    import numpy as np

    # Convert to HSV
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)  # OpenCV hue: 0..179

    # Build a binary mask for the hull
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Extract hue pixels inside hull
    hue_pixels = hue[mask > 0]
    if hue_pixels.size == 0:
        return None, 0

    # Convert OpenCV hue [0,179] to angle [0,2pi)
    angles = hue_pixels * (2.0 * np.pi / 180.0)

    # Circular mean
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    mean_angle = np.arctan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi

    # Convert back to OpenCV hue range [0,180)
    mean_hue_opencv = mean_angle * (180.0 / (2.0 * np.pi))

    return float(mean_hue_opencv), int(hue_pixels.size)
