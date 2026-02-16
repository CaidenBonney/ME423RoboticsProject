import pyrealsense2 as rs
import numpy as np
import cv2

# Find devices seen by realsense
ctx = rs.context()

# Get list of device names
camera_arr = [d.get_info(rs.camera_info.name) for d in ctx.query_devices()]
if camera_arr:
    print(f"Camera(s) found: {', '.join(x for x in camera_arr)}")
else:
    raise Exception("No cameras found")

# Start streaming pipeline and config object
pipeline = rs.pipeline()
config = rs.config()

# D435 streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Optional: align depth to color so pixels match
align = rs.align(rs.stream.color)

cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Get frameset
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        # Separate frames into depth and color
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # skip if missing frames received
        if not depth_frame or not color_frame:
            continue
        
        # Get data from frames as numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Depth is 16-bit; visualize it nicely:
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
