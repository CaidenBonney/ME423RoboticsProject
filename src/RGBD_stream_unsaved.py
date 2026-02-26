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
import math
import os

# show code is running
print("RUNNING...")
warmup_frames = 30
warmup_frames_count = 0
fps = 60
W, H = 640, 480
warm_up_video_path = "src/videos/warmup_video.mp4"

if os.path.exists(warm_up_video_path):
    try:
        os.remove(warm_up_video_path)
        print(f"Deleted old video: {warm_up_video_path}")
    except Exception as e:
        print(f"Could not delete video {warm_up_video_path}: {e}")
else:
    print(f"No existing video to delete at: {warm_up_video_path}")

# initialize background subtractor
cap = cv2.VideoCapture(3) # This number may be different for every machine. It corresponds to the port that the camera is attached to
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

bs = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=25, detectShadows=False
)
print(bs)


# Warm up background model
for i in range(warmup_frames):
    ret, frame = cap.read()
    if not ret:
        break
    bs.apply(frame, learningRate=0.05)
print("WARMED UP BACKGROUND MODEL...")


try:
    # --- White-ball HSV mask thresholds ---
    # White ≈ low saturation + high value. Tune V_low and S_high if needed.
    S_high = 70      # max saturation allowed (lower = stricter "white")
    V_low = 185      # min brightness allowed (higher = stricter "white")

    # --- Blob filters (tune) ---
    area_min = 80
    area_max = 8000
    circularity_min = 0.25
    solidity_min = 0.40
    aspect_max = 1.8    # reject elongated objects (screw)

    k5 = np.ones((5, 5), np.uint8)
    k3 = np.ones((3, 3), np.uint8)

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

    last_pts = []
    while True:
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
            # handle depth pipeline
            # depth_frames = depth_pipeline.wait_for_frames()
            depth = depth_frames.get_depth_frame()
            depth_timestamp = depth_frames.get_timestamp()
            depth_data = depth.get_data()
            # depth_image = np.asanyarray(depth.get_data())
            depth_image = np.asarray(depth.get_data(), dtype=np.uint8)
            depth_map = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            frames_count += 1
            # Our operations on the frame come here

        
            frame = rgb_image
            # --- White mask (shaded region) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # Hue doesn't matter for white; keep all hues
            white = cv2.inRange(
                hsv,
                (0, 0, V_low),          # low S, high V
                (179, S_high, 255)
            )

            # Clean up the mask (region-based approach)
            white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k5, iterations=1)
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k5, iterations=2)

            # --- Motion mask ---
            fg = bs.apply(frame, learningRate=0.002)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
            fg = cv2.dilate(fg, k5, iterations=1)

            # Only keep pixels that are BOTH white AND moving
            mask = cv2.bitwise_and(white, fg)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best = None

            # prediction: constant velocity from last two points
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
                aspect = max(w / max(1, h), h / max(1, w))  # >= 1

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

                dist = 0.0
                if pred is not None:
                    dist = math.hypot(cx - pred[0], cy - pred[1])

                # Score: prefer near prediction + round + solid + reasonable area
                score = (
                    -2.0 * dist
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

            out = frame.copy()

            if best is not None:
                cx, cy = best["cx"], best["cy"]
                last_pts.append((cx, cy))
                if len(last_pts) > 10:
                    last_pts = last_pts[-10:]

                # draw hull + center
                cv2.drawContours(out, [best["hull"]], -1, (0, 255, 0), 2)
                cv2.circle(out, (int(cx), int(cy)), 4, (255, 0, 0), -1)

                x, y, w, h = best["bbox"]
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.putText(out, f"area={int(best['area'])}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(out, f"circ={best['circularity']:.2f} sol={best['solidity']:.2f} asp={best['aspect']:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            #     records.append((frame_idx, cx, cy, best["area"], best["circularity"], best["solidity"], best["aspect"]))
            # else:
            #     records.append((frame_idx, np.nan, np.nan, 0, 0.0, 0.0, 0.0))

            writer.write(out)
            # frame_idx += 1

            cap.release()
            writer.release()


            # display rgb and depth frames
            # cv2.imshow('rgb_cam', rgb_image)
            cv2.imshow('rgb_cam', out)
            cv2.imshow('depth_cam', depth_map)
            # cv2.imshow('Canny Edges', edges)
        print(f"FRAME {frames_count} CAPTURED...{rgb_timestamp - start_time}")
        if cv2.waitKey(1) == ord('q'):
            break
    exit(0)
except Exception as e:
    print("FAILED")
    print(e)
    pass

