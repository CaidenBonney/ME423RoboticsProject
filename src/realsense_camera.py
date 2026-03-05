
"""
RealSense D400 camera wrapper for ME423 Robotics Project (cam branch).

Goals:
- Start/stop a single RealSense pipeline streaming synchronized RGB + depth.
- Align depth to the RGB frame so (u,v) from RGB detection can be used directly to read depth.
- Provide helpers to:
    * get color/depth frames (numpy)
    * detect the flying ball in RGB
    * robustly read depth at a pixel (median in a window)
    * convert (u,v,depth) to 3D in the CAMERA frame
    * transform CAMERA-frame points into the ROBOT BASE frame with a 4x4 T_base_cam

Dependencies:
    pip install pyrealsense2 opencv-python numpy

Notes:
- RealSense depth is returned in meters by depth_frame.get_distance(u,v).
- Alignment uses rs.align(rs.stream.color) so depth is in RGB pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError("pyrealsense2 is required. Install librealsense / pyrealsense2.") from e


@dataclass
class BallDetection:
    """Result of ball detection in the RGB image."""
    center_uv: Tuple[int, int]
    radius_px: float
    contour_area: float
    score: float  # higher is better
    mask: Optional[np.ndarray] = None  # optional debug mask


class RealSenseD400Camera:
    """
    Camera class that wraps a RealSense D400 stereo depth camera + RGB sensor.

    Typical usage:
        cam = RealSenseD400Camera(W=640, H=480, fps=60)
        cam.start()
        color, depth = cam.get_frames(aligned=True)
        det = cam.detect_ball(color)
        if det:
            z = cam.depth_at(*det.center_uv, depth_frame=depth, window=7)
            p_cam = cam.deproject_pixel(*det.center_uv, z)
            p_base = cam.to_base(p_cam)
        cam.stop()
    """

    def __init__(
        self,
        W: int = 640,
        H: int = 480,
        fps: int = 60,
        *,
        enable_color: bool = True,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        # 4x4 homogeneous transform from CAMERA frame to ROBOT BASE frame.
        # If you don't have this yet, leave as identity and update later with set_T_base_cam().
        T_base_cam: Optional[np.ndarray] = None,
        # Optional: whether to undistort the RGB image using OpenCV calibration (if you want)
        opencv_rgb_K: Optional[np.ndarray] = None,
        opencv_rgb_dist: Optional[np.ndarray] = None,
    ):
        self.W = int(W)
        self.H = int(H)
        self.fps = int(fps)
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color

        self.pipeline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None

        self.align = rs.align(rs.stream.color) if align_depth_to_color else None

        self.depth_scale: Optional[float] = None

        self.T_base_cam = np.eye(4, dtype=np.float64) if T_base_cam is None else np.array(T_base_cam, dtype=np.float64)
        self._assert_T(self.T_base_cam)

        self.opencv_rgb_K = None if opencv_rgb_K is None else np.array(opencv_rgb_K, dtype=np.float64)
        self.opencv_rgb_dist = None if opencv_rgb_dist is None else np.array(opencv_rgb_dist, dtype=np.float64).reshape(-1, 1)

        # cached intrinsics for deprojection (after alignment this should match color resolution)
        self._aligned_intrinsics: Optional[rs.intrinsics] = None

    # ------------------------- lifecycle -------------------------

    def start(self) -> None:
        """Start streaming."""
        if self.pipeline is not None:
            return

        self.pipeline = rs.pipeline()
        cfg = rs.config()

        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.fps)
        if self.enable_color:
            cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(cfg)

        # depth scale
        if self.enable_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())

        # warm up auto-exposure etc.
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # cache aligned intrinsics once we have aligned frames
        if self.enable_depth and self.enable_color and self.align is not None:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            # depth is now in the COLOR coordinate system
            self._aligned_intrinsics = depth.profile.as_video_stream_profile().intrinsics
        elif self.enable_depth:
            frames = self.pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            self._aligned_intrinsics = depth.profile.as_video_stream_profile().intrinsics

    def stop(self) -> None:
        """Stop streaming."""
        if self.pipeline is None:
            return
        self.pipeline.stop()
        self.pipeline = None
        self.profile = None
        self._aligned_intrinsics = None

    # ------------------------- frames -------------------------

    def get_frames(self, *, aligned: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Return (color_bgr, depth_frame_or_depth_meters_image).

        - color_bgr: HxWx3 uint8
        - depth: by default returns the *depth frame* (pyrealsense2 frame) so you can call get_distance(u,v)
          If you want an ndarray depth in meters, use get_depth_image_meters().
        """
        self._require_started()

        frames = self.pipeline.wait_for_frames()

        if aligned and self.enable_depth and self.enable_color and self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame() if self.enable_color else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None

        if color_frame is None:
            raise RuntimeError("Color stream not enabled / no color frame.")
        if depth_frame is None:
            raise RuntimeError("Depth stream not enabled / no depth frame.")

        color = np.asanyarray(color_frame.get_data())  # BGR

        if self.opencv_rgb_K is not None and self.opencv_rgb_dist is not None:
            color = cv2.undistort(color, self.opencv_rgb_K, self.opencv_rgb_dist)

        return color, depth_frame

    def get_depth_image_meters(self, depth_frame: Any) -> np.ndarray:
        """Convert a RealSense depth frame (z16) into a float32 depth image in meters."""
        if self.depth_scale is None:
            raise RuntimeError("Depth scale unknown; did you call start() with depth enabled?")
        z16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        return z16 * self.depth_scale

    # ------------------------- ball detection -------------------------

    def detect_ball(
        self,
        color_bgr: np.ndarray,
        *,
        # HSV thresholds for a "white" ball (tune for your lighting)
        S_high: int = 70,
        V_low: int = 185,
        # blob filters (tune)
        area_min: int = 80,
        area_max: int = 8000,
        circularity_min: float = 0.25,
        solidity_min: float = 0.40,
        aspect_max: float = 1.8,
        return_mask: bool = False,
    ) -> Optional[BallDetection]:
        """
        Detect the most ball-like white blob and return its center pixel in RGB coordinates.

        This is adapted from src/track_ball_center.py but made real-time and returns a single best candidate.
        """
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

        # white: low saturation, high value
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = ((s <= S_high) & (v >= V_low)).astype(np.uint8) * 255

        # cleanup
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[BallDetection] = None

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < area_min or area > area_max:
                continue

            peri = float(cv2.arcLength(cnt, True))
            if peri <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
            solidity = (area / hull_area) if hull_area > 1e-6 else 0.0

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = (w / h) if h > 0 else 999.0

            if circularity < circularity_min:
                continue
            if solidity < solidity_min:
                continue
            if aspect > aspect_max or (1.0 / aspect) > aspect_max:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # simple score: prefer larger, more circular blobs
            score = (area * circularity * solidity)

            cand = BallDetection(
                center_uv=(int(round(cx)), int(round(cy))),
                radius_px=float(radius),
                contour_area=area,
                score=float(score),
                mask=mask if return_mask else None,
            )

            if best is None or cand.score > best.score:
                best = cand

        return best

    # ------------------------- depth + 3D -------------------------

    def depth_at(
        self,
        u: int,
        v: int,
        *,
        depth_frame: Any,
        window: int = 7,
        invalid_value: float = 0.0,
    ) -> float:
        """
        Robust depth in meters at (u,v) by taking the median within a window.

        window: odd integer size (e.g., 1,3,5,7,11). window=1 reads exactly at (u,v).
        """
        if window <= 1:
            return float(depth_frame.get_distance(int(u), int(v)))

        if window % 2 == 0:
            window += 1

        half = window // 2
        us = range(max(0, u - half), min(self.W, u + half + 1))
        vs = range(max(0, v - half), min(self.H, v + half + 1))

        vals = []
        for vv in vs:
            for uu in us:
                z = float(depth_frame.get_distance(int(uu), int(vv)))
                if z > 0:
                    vals.append(z)

        if not vals:
            return float(invalid_value)

        return float(np.median(np.array(vals, dtype=np.float32)))

    def deproject_pixel(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """
        Convert aligned (u,v) + depth (meters) into a 3D point in the camera frame.

        Returns np.array([X, Y, Z]) in meters, using the standard RealSense camera frame:
            +X right, +Y down, +Z forward.
        """
        if self._aligned_intrinsics is None:
            raise RuntimeError("Intrinsics not available. Did you call start()?")

        pt = rs.rs2_deproject_pixel_to_point(self._aligned_intrinsics, [float(u), float(v)], float(depth_m))
        return np.array(pt, dtype=np.float64)

    def to_base(self, p_cam: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera frame to robot base frame using T_base_cam."""
        p = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
        out = self.T_base_cam @ p
        return out[:3]

    def set_T_base_cam(self, T_base_cam: np.ndarray) -> None:
        T = np.array(T_base_cam, dtype=np.float64)
        self._assert_T(T)
        self.T_base_cam = T

    # ------------------------- end-to-end helper -------------------------

    def get_ball_pose(
        self,
        *,
        depth_window: int = 7,
        detection_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience call:
        - grab aligned frames
        - detect ball center in RGB
        - read depth
        - compute 3D camera point and base point

        Returns dict with keys:
            color (np.ndarray), detection (BallDetection), depth_m, p_cam (np.ndarray), p_base (np.ndarray)
        or None if no detection / invalid depth.
        """
        detection_kwargs = detection_kwargs or {}

        color, depth_frame = self.get_frames(aligned=True)
        det = self.detect_ball(color, **detection_kwargs)
        if det is None:
            return None

        u, v = det.center_uv
        z = self.depth_at(u, v, depth_frame=depth_frame, window=depth_window)
        if z <= 0:
            return None

        p_cam = self.deproject_pixel(u, v, z)
        p_base = self.to_base(p_cam)

        return {
            "color": color,
            "depth_frame": depth_frame,
            "detection": det,
            "depth_m": z,
            "p_cam": p_cam,
            "p_base": p_base,
        }

    # ------------------------- utils -------------------------

    def _require_started(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Camera not started. Call start() first.")

    @staticmethod
    def _assert_T(T: np.ndarray) -> None:
        if T.shape != (4, 4):
            raise ValueError("T_base_cam must be 4x4 homogeneous matrix.")
        if not np.isfinite(T).all():
            raise ValueError("T_base_cam contains non-finite values.")
