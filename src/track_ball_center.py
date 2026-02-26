################################################################################
# IF RERUN, DELETE THE CSV FILES BEFORE RUNNING THE TRACKER ####################
# MAIN BLOCK AT BOTTOM TO RUN THE TRACKER AND SAVE RESULTS  ####################
################################################################################

import cv2
import numpy as np
import pandas as pd
import math
import os


def track_white_ball(
    in_path: str = "",
    out_video_path: str = "",
    out_csv_path: str = "",
    warmup_frames: int = 60,

    # --- White-ball HSV mask thresholds ---
    # White ≈ low saturation + high value. Tune V_low and S_high if needed.
    S_high: int = 70,      # max saturation allowed (lower = stricter "white")
    V_low: int = 185,      # min brightness allowed (higher = stricter "white")

    # --- Blob filters (tune) ---
    area_min: int = 80,
    area_max: int = 8000,
    circularity_min: float = 0.25,
    solidity_min: float = 0.40,
    aspect_max: float = 1.8,     # reject elongated objects (screw)
    delete_old_csv: bool = True,
    delete_old_vid = True

):
    # Raise error if input paths are empty
    if in_path == "" or out_video_path == "" or out_csv_path == "":
        raise RuntimeError("Must provide in_path, out_video_path, and out_csv_path")
    
    # --- Delete old output files if requested ---
    if delete_old_vid:
        if os.path.exists(out_video_path):
            try:
                os.remove(out_video_path)
                print(f"Deleted old video: {out_video_path}")
            except Exception as e:
                print(f"Could not delete video {out_video_path}: {e}")
        else:
            print(f"No existing video to delete at: {out_video_path}")
    if delete_old_csv:
        if os.path.exists(out_csv_path):
            try:
                os.remove(out_csv_path)
                print(f"Deleted old CSV: {out_csv_path}")
            except Exception as e:
                print(f"Could not delete CSV {out_csv_path}: {e}")
        else:
            print(f"No existing CSV to delete at: {out_csv_path}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (W, H),
        True,
    )


    # Background subtractor to remove static bright objects (like the screw)
    bs = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=False
    )

    k5 = np.ones((5, 5), np.uint8)
    k3 = np.ones((3, 3), np.uint8)

    # Warm up background model
    records = []
    for i in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bs.apply(frame, learningRate=0.05)
        writer.write(frame)
        records.append((i, np.nan, np.nan, 0, 0.0, 0.0, 0.0))

    last_pts = []
    frame_idx = len(records)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # --- White mask (shaded region) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

            records.append((frame_idx, cx, cy, best["area"], best["circularity"], best["solidity"], best["aspect"]))
        else:
            records.append((frame_idx, np.nan, np.nan, 0, 0.0, 0.0, 0.0))

        writer.write(out)
        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(records, columns=["frame", "cx", "cy", "area", "circularity", "solidity", "aspect"])
    df["time_s"] = df["frame"] / fps
    df.to_csv(out_csv_path, index=False)
    return df


if __name__ == "__main__":
    track_white_ball(in_path = "src/videos/rgb_video_arc.mp4",
    out_video_path = "src/videos/tracked_ball_arc.mp4",
    out_csv_path = "src/videos/ball_centers_arc.csv",
    warmup_frames = 60,
    # --- White-ball HSV mask thresholds ---
    S_high = 70,      
    V_low = 185, 
    # --- Blob filters (tune) ---
    area_min = 80,
    area_max = 8000,
    circularity_min = 0.25,
    solidity_min = 0.40,
    aspect_max = 1.8,
    delete_old_csv = True,
    delete_old_vid = True)

    track_white_ball(in_path = "src/videos/rgb_video_drop.mp4",
    out_video_path = "src/videos/tracked_ball_drop.mp4",
    out_csv_path = "src/videos/ball_centers_drop.csv",
    warmup_frames = 20,
    # --- White-ball HSV mask thresholds ---
    S_high = 70,      
    V_low = 185, 
    # --- Blob filters (tune) ---
    area_min = 80,
    area_max = 8000,
    circularity_min = 0.25,
    solidity_min = 0.40,
    aspect_max = 1.8,
    delete_old_csv = True,
    delete_old_vid = True)