## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 RealSense, Inc. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# show code is running
print(f"Running...{time.time()}")

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streamss
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streamingl
pipeline.start(config)

cap = cv2.VideoCapture(3) # This number may be different for every machine. It corresponds to the port that the camera is attached to
# capture 10 images!

#index = 3 for lab computers
for i in range(10):
    ret, frame = cap.read()
    if ret:
        # Display the frame using imshow
        cv2.imshow("Captured Frame", frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window
    else:
        print("Error: Could not capture a frame.")




# show video stream
try:
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue
        # depth_data = depth.get_data()
        depth_image = np.asanyarray(depth.get_data())
        cv2.imshow('RealSense', depth_image)
    exit(0)
except rs.error as e:
   # Method calls against librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
   print("    %s\n", e.what())
   exit(1)
except Exception as e:
    print(e)
    pass