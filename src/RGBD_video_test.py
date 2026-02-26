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
print("RUNNING...")
# print(f"Running...{time.time()}")

# Create a context object. This object owns the handles to all connected realsense devices
# pipeline = rs.pipeline()

# # Configure streamss
# depth_config = rs.config()
# depth_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # Start streamingl
# pipeline.start(depth_config)

# cap = cv2.VideoCapture(2) # This number may be different for every machine. It corresponds to the port that the camera is attached to
# # capture 10 images!

# #index = 3 for lab computers
# # for i in range(10):
# #     ret, frame = cap.read()
# #     if ret:
# #         # Display the frame using imshow
# #         cv2.imshow("Captured Frame", frame)
# #         cv2.waitKey(0)  # Wait for a key press to close the window
# #         cv2.destroyAllWindows()  # Close the window
# #     else:
# #         print("Error: Could not capture a frame.")
 
# show video stream
try:
    # cv2.VideoWriter(filename, fourcc, fps, frame_size, isColor)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    depth_video_writer = cv2.VideoWriter("src/videos/depth_video_arc.mp4", fourcc, 60, (640, 480), False)
    raw_video_writer = cv2.VideoWriter("src/videos/rgb_video_arc.mp4", fourcc, 60, (640, 480), True)
    annotated_video_writer = cv2.VideoWriter("src/videos/annotated_video_arc.mp4", fourcc, 60, (640, 480), True)
    canny_video_writer = cv2.VideoWriter("src/videos/canny_video_arc.mp4", fourcc, 60, (640, 480), False)
    # Create a context object. This object owns the handles to all connected realsense devices
    rgb_pipeline = rs.pipeline()
    depth_pipeline = rs.pipeline()

    # Configure streams
    depth_config = rs.config()
    depth_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    rgb_config = rs.config()
    rgb_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Start streaming
    rgb_pipeline.start(rgb_config)
    depth_pipeline.start(depth_config)
    cv2.namedWindow('depth_cam', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('rgb_cam', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Canny Edges', cv2.WINDOW_AUTOSIZE)
    frames_count = 0
    start_time = time.time()
    while frames_count < 900:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        (rgb_frame_present, rgb_frames) = rgb_pipeline.try_wait_for_frames()
        (depth_frame_present, depth_frames) = depth_pipeline.try_wait_for_frames()
        if rgb_frame_present and depth_frame_present:
            rgb_timestamp = time.time()
            depth_timestamp = time.time()
            # rgb_frames = rgb_pipeline.wait_for_frames()
            rgb = rgb_frames.get_color_frame()
            rgb_timestamp = rgb_frames.get_timestamp()
            rgb_image = np.asanyarray(rgb.get_data())
            raw_video_writer.write(rgb_image)
            frames_count += 1
            # Our operations on the frame come here
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            # 2. Optional: Apply Gaussian blur to reduce noise
            # The Canny function does internal blurring, but extra can help for noisy images
            gray = hsv[:,:,2]
            blurred = cv2.GaussianBlur(gray, (5, 5), 1)
            # 3. Apply the Canny edge detector
            # Common practice is to use a 2:1 or 3:1 ratio for the thresholds (e.g., 50 and 150)
            edges = cv2.Canny(blurred, 120, 170) # The output is a binary image (edge map)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ball_contour = 0
            shape_ratio = 0
            area = 0
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                perimeter = cv2.arcLength(contours[i], False)
                shape_ratio = (perimeter ** 2)/area if area != 0 else 0
                if shape_ratio > 10 and shape_ratio < 15 and area > 1000 and area < 5000:
                    ball_contour = i
                    break
            # handle depth pipeline
            # depth_frames = depth_pipeline.wait_for_frames()
            depth = depth_frames.get_depth_frame()
            depth_timestamp = depth_frames.get_timestamp()
            depth_data = depth.get_data()
            # depth_image = np.asanyarray(depth.get_data())
            depth_image = np.asarray(depth.get_data(), dtype=np.uint8)
            depth_map = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            
            # display rgb and depth frames
            cv2.drawContours(image=rgb_image, contours=contours, contourIdx=ball_contour, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(rgb_image, f"Shape Ratio: {shape_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(rgb_image, f"Area: {area}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('rgb_cam', rgb_image)
            cv2.imshow('depth_cam', depth_map)
            cv2.imshow('Canny Edges', edges)
            annotated_video_writer.write(rgb_image)
            canny_video_writer.write(edges)
            depth_video_writer.write(depth_image)
        if cv2.waitKey(1) == ord('q'):
            break
        print(f"FRAME {frames_count} CAPTURED...{rgb_timestamp - start_time}")
        frames_count += 1

    exit(0)
    raw_video_writer.release()
    depth_video_writer.release()
    annotated_video_writer.release()
    canny_video_writer.release()
except Exception as e:
    print("FAILED")
    print(e)
    pass