import pyrealsense2 as rs
import numpy as np
import cv2
import time

cap = cv2.VideoCapture("src\\videos\\rgb_video_2.mp4") # This number may be different for every machine. It corresponds to the port that the camera is attached to
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # 2. Optional: Apply Gaussian blur to reduce noise
    # The Canny function does internal blurring, but extra can help for noisy images
    gray = hsv[:,:,2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.5)

    # 3. Apply the Canny edge detector
    # Common practice is to use a 2:1 or 3:1 ratio for the thresholds (e.g., 50 and 150)
    edges = cv2.Canny(blurred, 50, 80) # The output is a binary image (edge map)

    # 4. Display the results
    cv2.imshow('Canny Edges', edges)
    cv2.imshow('frame', hsv[:, :, 0])
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()