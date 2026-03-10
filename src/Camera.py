import os
from re import L
import time
from typing import Optional
import numpy as np
from numpy.random import randint
import cv2
import pyrealsense2 as rs

class Camera:
    def __init__(self) -> None:
        self.startTime = time.time()
        self.sampleRate = 30
        self.sampleTime = 1 / self.sampleRate
        self.bs = None # background susbtractor mpdel
        self.intrinsics = None # camera intrinsics
        self.robotTransformation = None # transformation from camera frame to robot frame
        self.pipeline = None # realsense pipeline
        self.profile = None # realsense pipeline profile
        self.align = None # realsense align object
        self.cameraPortID = 3 # This number may be different for every machine. It corresponds to the port that the camera is attached to
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cam_setup()

    def elapsed_time(self) -> float:
        return time.time() - self.startTime
    
    def capture_image(self) -> rs.align:
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
        self.camera_matrix = np.array(([800, 0, 320], [0, 800, 240],[0, 0, 1]), dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.create_background_model()
        print("BACKGROUND MODEL CREATED...")
        print("SOLVING ROBOT TRANSFORMATION...")
        self.get_robot_transformation()
        print("ROBOT TRANSFORMATION SOLVED...")

    def get_robot_transformation(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        rgb_frames = aligned_frames.get_color_frame()
        depth_frames = aligned_frames.get_depth_frame()
        # PROGRAM FAILS TO MAKE IT PAST THIS POINT, LIKELY DUE TO SOME ISSUE WITH THE REALSENSE PIPELINE OR ALIGNMENT. NEED TO DEBUG.

        aligned_frames = self.capture_image()
        rgb = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        frame = np.asanyarray(rgb.get_data())
        CALIBRATION_FILE = "camera_calib.yml"   # <-- path to your calibration YAML
        # MARKER_ID = [67, 1, 2, 3, 5, 4, 6]
        MARKER_ID = 67  
        MARKER_LENGTH = 0.07  # marker side length in meters (change to yours)
        CAMERA_INDEX = self.cameraPortID
        ARUCO_DICT = cv2.aruco.DICT_4X4_250
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
        object_points = create_marker_object_points(MARKER_LENGTH)
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector_params = cv2.aruco.DetectorParameters()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect markers
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            ids = ids.flatten()
            print("Detected marker IDs:", ids)
            for i, marker_id in enumerate(ids):
                if marker_id == MARKER_ID:
                    image_points = corners[i].reshape(4, 2)
                    # print(image_points)
                    # Estimate marker pose w.r.t camera
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,  
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if success:
                        marker_center = image_points.mean(axis=0)  # (u,v) center of the marker in pixel coordinates
                        # print("marker id: ", marker_id)
                        # print(marker_centers)
                        depth_meters = depth.get_distance(int(marker_center[0]), int(marker_center[1]))
                        real_point = rs.rs2_deproject_pixel_to_point(self.intrinsics, marker_center, depth_meters)
                        real_point = [-1*coord for coord in real_point]
                        # Invert to get camera pose w.r.t marker
                        rvec_cam, tvec_cam = invert_pose(rvec, tvec)
                        tvec_cam = np.asanyarray(tvec_cam).reshape(3,1)  # ensure tvec_cam is a 1D array of shape (3,)
                        R_cam, _ = cv2.Rodrigues(rvec_cam)
                        T_cam_in_marker = np.eye(4)
                        T_cam_in_marker[:3, :3] = R_cam
                        T_cam_in_marker[:3, 3] = tvec_cam.reshape(3)
                        T_marker_in_cam = np.eye(4)
                        R_marker_in_cam = np.linalg.inv(R_cam)
                        T_marker_in_cam[:3, :3] = R_marker_in_cam
                        T_marker_in_cam[:3, 3] = np.asanyarray(real_point).reshape(3,1)
                        T_cam_in_marker = np.linalg.inv(T_marker_in_cam)
                        # Draw results for visualization
                        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        # cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.75)
                        # print("\n=== Marker 67 Detected ===")
                        # print("Camera Position in Marker Frame (meters):")
                        # print(tvec_cam.reshape(3))
                        # print("Camera Rotation Vector (Rodrigues, radians):")
                        # print(rvec_cam.reshape(3))
                        # print("4x4 Transformation Matrix:")
                        # print(T_cam_in_marker)
                        return T_cam_in_marker
                    else:
                        print(f"Failed to estimate pose for marker ID {marker_id}")
                        return np.eye(4)  # return identity if pose estimation fails
        pass

    def create_background_model(self, warm_up_video_path: str = "src/videos/warmup_video.mp4", warmup_frames: int = 30, fps: int = 60, W: int = 640, H: int = 480) -> None:
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
        cap = cv2.VideoCapture(self.cameraPortID) # This number may be different for every machine. It corresponds to the port that the camera is attached to
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
        cap.release
        # Background subtractor to remove static bright objects (like the screw)
        self.bs = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        # Warm up background model
        for i in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.bs.apply(frame, learningRate=0.05)
        print("WARMED UP BACKGROUND MODEL...")
    pass

    def image_processing(self, aligned_frames: rs.align) -> np.ndarray:
        # TODO: convert image to XYZ
        rgb = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        depth_data = depth.get_data()
        depth_image = np.asarray(depth.get_data(), dtype=np.uint8)
        # depth_map = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        pass
    
    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self) -> tuple[Optional[np.ndarray], Optional[np.float64], Optional[np.ndarray]]:
        input = self.capture_image()
        # XYZ = self.image_processing(input)
        
        XYZ = np.random.uniform(
            low=np.array([-0.50, -0.10, -0.55], dtype=np.float64),
            high=np.array([-0.40, 0.10, -0.45], dtype=np.float64),
        )
        
        gripper_Cmd = None
        led_Cmd = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        
        return XYZ, gripper_Cmd, led_Cmd
