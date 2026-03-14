# from multiprocessing import Value
import queue
import cv2
import numpy as np
import threading
import time

from Arm import Arm
from Camera import Camera

ballXYZ_queue: queue.Queue = queue.Queue(maxsize=1000)
cam = Camera()

# window_name = "camera_pov"
# window = cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
# cv2.imshow(window_name, cam.current_frame)
while True:
    start = cam.elapsed_time()
    ballXYZ = cam.capture_and_process()
    # start = time.perf_counter()
    # ballXYZ = cam.capture_and_process()
    # elapsed = time.perf_counter() - start
    # print(f"capture_and_process took {elapsed:.6f} seconds ({elapsed*1000:.2f} ms)")



    # Only publish if the camera produced a valid command
    try:
        
        cv2.waitKey(1) # wait 1 ms. needed to display video feed
        if ballXYZ_queue.full():
            ballXYZ_queue.get_nowait()
        ballXYZ_queue.put_nowait(ballXYZ)
        # start_show = time.perf_counter()
        cam.show_image()
        # elapsed_show = time.perf_counter() - start_show
        # print(f"im_show took {elapsed_show:.6f} seconds ({elapsed_show*1000:.2f} ms)")

        # print("imshowing frame in tester...")
        print(ballXYZ)
    except queue.Empty:
        pass


