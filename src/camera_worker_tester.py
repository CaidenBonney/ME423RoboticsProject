# from multiprocessing import Value
import queue
import numpy as np
import threading
import time

from Arm import Arm
from Camera import Camera

ballXYZ_queue: queue.Queue = queue.Queue(maxsize=1000)
cam = Camera()

while True:
    start = cam.elapsed_time()
    ballXYZ = cam.capture_and_process()

    # Only publish if the camera produced a valid command
    try:
        if ballXYZ_queue.full():
            ballXYZ_queue.get_nowait()
        ballXYZ_queue.put_nowait(ballXYZ)
        print(ballXYZ)
    except queue.Empty:
        pass


