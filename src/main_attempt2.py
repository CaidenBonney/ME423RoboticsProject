# from multiprocessing import Value
import queue
import cv2
import numpy as np
import threading
import time

from Arm import Arm
from Camera import Camera


def camera_worker(
    cam: Camera,
    ballXYZ_queue: queue.Queue,
    stop_event: threading.Event,
    ready: threading.Event
) -> None:
    ready.set()

    start = cam.elapsed_time()
    while not stop_event.is_set():
        ballXYZ = cam.capture_and_process()

        if ballXYZ is None:
            continue

        try:
            if ballXYZ_queue.full():
                ballXYZ_queue.get_nowait()
            ballXYZ_queue.put_nowait(ballXYZ)
        except queue.Empty:
            pass

        # # Pause/sleep to maintain same rate
        # sleep_time = cam.sampleTime - (cam.elapsed_time() - start) % cam.sampleTime
        # if sleep_time > 0:
        #     time.sleep(sleep_time)

    pass


def camera_display_worker(
    cam: Camera,
    stop_event: threading.Event,
    ready: threading.Event
) -> None:
    ready.set()

    while not stop_event.is_set():
        cv2.imshow("camera_pov", cam.current_frame)
        cv2.waitKey(1)
        time.sleep(0.001)

    cv2.destroyWindow("camera_pov")


def arm_display_worker(
    arm_frame_shared: dict,
    stop_event: threading.Event,
    ready: threading.Event
) -> None:
    ready.set()

    while not stop_event.is_set():
        frame = arm_frame_shared["frame"]
        if frame is not None:
            cv2.imshow("arm_pov", frame)
        cv2.waitKey(1)
        time.sleep(0.001)

    cv2.destroyWindow("arm_pov")


def arm_worker(
    cam: Camera,
    arm_frame_shared: dict,
    ballXYZ_queue: queue.Queue,
    stop_event: threading.Event,
    ready: threading.Event
) -> None:
    arm = Arm()
    ready.set()

    moved = False
    start = arm.elapsed_time()

    while not stop_event.is_set() and arm.myArm.status:
        if ballXYZ_queue.empty():
            continue

        ballXYZ = ballXYZ_queue.get_nowait()

        try:
            # Grab the latest camera frame inside the arm thread
            arm_frame = cam.current_frame.copy()

            # Draw whatever the arm thread wants onto its own frame copy
            cv2.putText(
                arm_frame,
                f"ballXYZ: [{ballXYZ[0]:.3f}, {ballXYZ[1]:.3f}, {ballXYZ[2]:.3f}]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            phi_cmd = arm.ballXYZ_to_phi_cmd(ballXYZ)

            cv2.putText(
                arm_frame,
                f"phi_cmd: [{phi_cmd[0]:.3f}, {phi_cmd[1]:.3f}, {phi_cmd[2]:.3f}, {phi_cmd[3]:.3f}]",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            arm_frame_shared["frame"] = arm_frame

            if not moved:
                arm.move(phi_Cmd=phi_cmd)
                moved = True

        except ValueError as e:
            print(f"Command error: {e}")
            arm.home()

        if (arm.elapsed_time() - start) > arm.sampleTime and moved:
            moved = False
            start = start + arm.sampleTime

    try:
        arm.myArm.terminate()
    except Exception as e:
        print(f"Terminate error: {e}")


def main() -> None:
    stop_event = threading.Event()
    ballXYZ_queue: queue.Queue = queue.Queue(maxsize=1000)

    cam = Camera()
    arm_frame_shared = {
        "frame": np.zeros_like(cam.current_frame)
    }

    cam_ready = threading.Event()
    cam_display_ready = threading.Event()
    arm_display_ready = threading.Event()
    arm_ready = threading.Event()

    cam_thread = threading.Thread(
        target=camera_worker,
        args=(cam, ballXYZ_queue, stop_event, cam_ready),
        daemon=False
    )

    cam_display_thread = threading.Thread(
        target=camera_display_worker,
        args=(cam, stop_event, cam_display_ready),
        daemon=False
    )

    arm_display_thread = threading.Thread(
        target=arm_display_worker,
        args=(arm_frame_shared, stop_event, arm_display_ready),
        daemon=False
    )

    arm_thread = threading.Thread(
        target=arm_worker,
        args=(cam, arm_frame_shared, ballXYZ_queue, stop_event, arm_ready),
        daemon=False
    )

    cam_thread.start()
    cam_display_thread.start()
    arm_display_thread.start()
    arm_thread.start()

    try:
        while arm_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cam_thread.join()
        cam_display_thread.join()
        arm_display_thread.join()
        arm_thread.join()


if __name__ == "__main__":
    main()