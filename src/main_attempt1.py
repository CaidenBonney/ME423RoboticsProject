import queue
import threading
import time
import numpy as np

from Arm import Arm
from Camera import Camera


def camera_worker(cmd_queue: queue.Queue, stop_event: threading.Event, ready: threading.Event) -> None:
    cam = Camera()
    ready.set()

    start = cam.elapsed_time()
    while not stop_event.is_set():
        cmd = cam.capture_and_process()

        # keep only newest
        try:
            if cmd_queue.full():
                cmd_queue.get_nowait()
            cmd_queue.put_nowait(cmd)
        except queue.Empty:
            pass

        sleep_time = cam.sampleTime - (cam.elapsed_time() - start) % cam.sampleTime
        if sleep_time > 0:
            time.sleep(sleep_time)

    # If you have a proper camera shutdown, do it here:
    # cam.close()


def arm_worker(cmd_queue: queue.Queue, stop_event: threading.Event, ready: threading.Event) -> None:
    arm = Arm()         # <-- create in SAME thread that uses it
    ready.set()

    latest_cmd = (
        np.array([-0.45, 0.0, -0.49], dtype=np.float64),
        np.float64(0.0),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )

    start = arm.elapsed_time()
    try:
        while not stop_event.is_set() and arm.myArm.status:
            try:
                latest_cmd = cmd_queue.get_nowait()
            except queue.Empty:
                pass

            XYZ, gripper_cmd, led_cmd = latest_cmd

            try:
                phi_cmd, _, _ = arm.XYZ_to_phi_cmd(XYZ)
                arm.move(phi_Cmd=phi_cmd, gripper_Cmd=gripper_cmd, led_Cmd=led_cmd)
            except ValueError as e:
                print(f"Command error: {e}")
                arm.home()

            sleep_time = arm.sampleTime - (arm.elapsed_time() - start) % arm.sampleTime
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # terminate only after loop ends and no more read_write_std calls can happen
        try:
            arm.myArm.terminate()
        except Exception as e:
            print(f"Terminate error: {e}")


def main() -> None:
    stop_event = threading.Event()
    cmd_queue: queue.Queue = queue.Queue(maxsize=1)

    cam_ready = threading.Event()
    arm_ready = threading.Event()

    cam_thread = threading.Thread(target=camera_worker, args=(cmd_queue, stop_event, cam_ready), daemon=False)
    arm_thread = threading.Thread(target=arm_worker, args=(cmd_queue, stop_event, arm_ready), daemon=False)

    cam_thread.start()
    arm_thread.start()

    # Wait for both to initialize (prevents “use before init” issues)
    cam_ready.wait()
    arm_ready.wait()

    try:
        while arm_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # unblock if needed (optional)
        # cmd_queue.put_nowait(latest_cmd)  # or just ignore
        cam_thread.join()
        arm_thread.join()


if __name__ == "__main__":
    main()