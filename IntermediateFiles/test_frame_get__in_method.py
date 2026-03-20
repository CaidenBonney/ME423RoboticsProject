import time
import cv2
import pyrealsense2 as rs
import numpy as np
class test_model:
    def __init__(self, name):
        self.name = name

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

    def capture_image(self) -> rs.align:
        frames = self.pipeline.wait_for_frames()
        return self.align.process(frames)
    
obj = test_model("test")
obj.cam_setup()
cv2.namedWindow('depth_cam', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('rgb_cam', cv2.WINDOW_AUTOSIZE)
while True:
    aligned_frames = obj.capture_image()
    depth_frame = aligned_frames.get_depth_frame().get_data()
    color_frame = aligned_frames.get_color_frame().get_data()
    cv2.imshow('depth_cam', np.asanyarray(depth_frame))
    cv2.imshow('rgb_cam', np.asanyarray(color_frame))
    if cv2.waitKey(1) == ord('q'):
        break
    print("Captured aligned frames at time: ", time.time() - obj.startTime)
