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

# ---------------- USER CONFIG ----------------
MARKER_ID = 67
MARKER_LENGTH_M = 0.07
ARUCO_DICT = cv2.aruco.DICT_4X4_250

W, H, FPS = 640, 480, 30

NEIGHBOR_RADIUS_PX = 2  # depth sampling neighborhood for marker corners
BALL_DEPTH_RADIUS_PX = 2  # depth sampling neighborhood for ball center
MIN_VALID_CORNERS = 3

# Ball color options
WHITE_BALL_COLOR = 0
ORANGE_BALL_COLOR = 1
GREEN_BALL_COLOR = 2

# Ball detector thresholds (tune if needed)
S_HIGH = 70
V_LOW = 185
AREA_MIN = 0
AREA_MAX = 12000
CIRC_MIN = 0.25
SOLID_MIN = 0.40
ASPECT_MAX = 1.8

WARMUP_FRAMES = 30

def detect_ball_center(frame_bgr, bs, last_pts, ball_color: int = WHITE_BALL_COLOR, using_bg_sub: bool = True):
    """ Detects the ball center in the frame using the frame, background subtractor and last detected points.
    
    Args:
        frame_bgr (np.ndarray): The frame to detect the ball center in.
        bs (cv2.BackgroundSubtractorMOG2): The background subtractor to use.
        last_pts (list): The last detected points.
        ball_color (int): The color of the ball to detect. Use the constants WHITE_BALL_COLOR, ORANGE_BALL_COLOR, or GREEN_BALL_COLOR.

    Returns:
      (found: Boolean, Optional[(u,v, best: dict)], "dict(mask=mask)") """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    k5 = np.ones((5, 5), np.uint8)
    # select ball color
    if ball_color == WHITE_BALL_COLOR:
        color_mask = cv2.inRange(hsv, (0, 0, V_LOW), (179, S_HIGH, 255))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    elif ball_color == ORANGE_BALL_COLOR:
    # Orange mask (covers the usual orange hue range; tune if needed)
        # lower_orange = (8, 160, 175)
        # upper_orange = (12, 255, 255)
        # lower_orange = (6, 120, 140)
        # upper_orange = (16, 255, 255)
        # lower_orange = (8, 150, 120)
        # upper_orange = (18, 255, 255)

        # Bright Indoor lighting
        lower_orange = (7, 160, 130)
        upper_orange = (18, 255, 255)

        # "Normal" Indoor Lighting
        # lower_orange = (7, 140, 110)
        # upper_orange = (18, 255, 255)

        # Dimmer lighting
        # lower_orange = (6, 130, 90)
        # upper_orange = (20, 255, 255)
        color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    elif ball_color == GREEN_BALL_COLOR:
        # Green mask (covers the usual green hue range; tune if needed)
        # lower_green = (92, 110, 120)
        # upper_green = (113, 255, 255)

        # More "robust" for changing lighting
        # lower_green = (38, 90, 100)
        # upper_green = (65, 255, 255)

        # sc of green ball
        # lower_green = (70, 140, 120)
        # upper_green = (82, 255, 255)
        lower_green = (70, 50, 40)
        upper_green = (90, 255, 255)
        using_bg_sub = True # bg sub seems to hurt green ball detection, so disable for green ball

        # sc of ball: robust for lighting
        # lower_green = (68, 120, 100)
        # upper_green = (84, 255, 255)

        color_mask = cv2.inRange(hsv, lower_green, upper_green)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k5, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    else:
        raise ValueError(f"Invalid ball color: {ball_color}")

    if using_bg_sub:
        k3 = np.ones((3, 3), np.uint8)

        fg = bs.apply(frame_bgr, learningRate=0.002)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
        fg = cv2.dilate(fg, k5, iterations=1)
        mask = cv2.bitwise_and(fg, color_mask)
    else:
        mask = color_mask
        
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

        # score = (-2.0 * dist_pred) + (0.25 * area) + (350.0 * circ) + (250.0 * solid) - (60.0 * aspect)
        score = (-5.0 * dist_pred) + (350.0 * circ) + (250.0 * solid) - (60.0 * aspect)


        if best is None or score > best["score"]:
            best = dict(score=score, cx=cx, cy=cy, hull=hull, bbox=(x, y, w, h))

    if best is None:
        return False, None, dict(mask=mask)

    u, v = int(round(best["cx"])), int(round(best["cy"]))
    last_pts.append((best["cx"], best["cy"]))
    if len(last_pts) > 10:
        last_pts[:] = last_pts[-10:]
    return True, (u, v, best), dict(mask=mask)

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
 
    # configure depth and color streamss
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    profile = pipeline.start(cfg)

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("ALIGN TYPE: ", type(align))

    cv2.namedWindow('depth_cam', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('rgb_cam', cv2.WINDOW_AUTOSIZE)
    frames_count = 0
    start_time = time.time()

    last_pts = []

    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print("Camera intrinsics:", intrinsics)
    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        rgb_frames = aligned_frames.get_color_frame()
        depth_frames = aligned_frames.get_depth_frame()
        
        # if rgb_frame_present and depth_frame_present:
        if rgb_frames and depth_frames:
            rgb_timestamp = time.time()
            depth_timestamp = time.time()
            frame = np.asanyarray(rgb_frames.get_data())
            # print("Ball center: ", detect_ball_center(np.asarray(rgb_frames.get_data()), bs, last_pts))
            found_ball, ball_info, mask = detect_ball_center(frame, bs, last_pts, ball_color=GREEN_BALL_COLOR)
            if found_ball:
                print("BALL DETECTED ...")
                u, v, best = ball_info
                cv2.circle(frame, (u, v), 5, (255, 0, 0), -1)
                cv2.drawContours(frame, [best["hull"]], -1, (0, 255, 0), 2)

                z = robust_depth_at_pixel(depth_frames, u, v, BALL_DEPTH_RADIUS_PX)
                print(f"Ball center (u,v): ({u}, {v}), depth: {z:.3f} m")
            else:   
                print("BALL NOT DETECTED ...")
            point = np.asarray([0, 0, 0], dtype=np.float64)
            cv2.imshow('rgb_cam', np.asanyarray(frame))
        if cv2.waitKey(1) == ord('q'):
            break
    exit(0)
except Exception as e:
    print("FAILED")
    print(e)
    pass
