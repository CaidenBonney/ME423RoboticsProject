import queue
import threading
import time

import numpy as np

from Arm import Arm
from Camera import Camera


def camera_loop(cam: Camera, cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    start = cam.elapsed_time()
    while not stop_event.is_set():
        cmd = cam.capture_and_process()

        # Keep only the newest camera-derived command.
        try:
            if cmd_queue.full():
                cmd_queue.get_nowait()
            cmd_queue.put_nowait(cmd)
        except queue.Empty:
            pass

        time.sleep(cam.sampleTime)

        # Pause/sleep to maintain same rate
        sleep_time = cam.sampleTime - (cam.elapsed_time() - start) % cam.sampleTime
        if sleep_time > 0:
            time.sleep(sleep_time)


def arm_loop(arm: Arm, cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    # initialize latest command to be sent to arm to home position with no gripper or LED activation
    latest_cmd = (
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.float64(0.0),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )

    start = arm.elapsed_time()
    while not stop_event.is_set() and arm.myArm.status:
        # get latest command from queue
        try:
            latest_cmd = cmd_queue.get_nowait()
        except queue.Empty:
            pass

        # send latest command to arm
        phi_cmd, gripper_cmd, led_cmd = latest_cmd
        try:
            arm.move(phi_Cmd=phi_cmd, gripper_Cmd=gripper_cmd, led_Cmd=led_cmd)
        except ValueError as e:
            print(f"Command error: {e}")
            arm.home()  # Move to home position on error

        # Pause/sleep to maintain same rate
        sleep_time = arm.sampleTime - (arm.elapsed_time() - start) % arm.sampleTime
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    stop_event = threading.Event()  # Main stop event for all threads

    # Shared queues for threads to communicate
    cmd_queue: queue.Queue = queue.Queue(maxsize=1)  # shared queue for arm and camera commands

    # Initialize arm and camera objects
    arm = Arm()
    cam = Camera()

    # Create and start threads for camera and arm
    cam_thread = threading.Thread(target=camera_loop, args=(cam, cmd_queue, stop_event), daemon=True)
    arm_thread = threading.Thread(target=arm_loop, args=(arm, cmd_queue, stop_event), daemon=True)
    # TODO: add thread for trajectory planning to implement between camera and arm threads if needed
    cam_thread.start()
    arm_thread.start()

    try:
        while arm_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cam_thread.join(timeout=1.0)
        arm_thread.join(timeout=1.0)
        arm.myArm.terminate()


if __name__ == "__main__":
    main()
