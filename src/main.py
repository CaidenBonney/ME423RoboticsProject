# from multiprocessing import Value
import queue
import numpy as np
import threading
import time

from Arm import Arm
from Camera import Camera


def camera_worker(ballXYZ_queue: queue.Queue, stop_event: threading.Event, ready: threading.Event) -> None:
    # cam = Camera()
    ready.set()
    return

    start = cam.elapsed_time()
    while not stop_event.is_set():
        ballXYZ = cam.capture_and_process()

        # Only publish if the camera produced a valid command
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

    # If you have a proper camera shutdown, do it here:
    # cam.close()
    pass


def arm_worker(ballXYZ_queue: queue.Queue, stop_event: threading.Event, ready: threading.Event) -> None:
    arm = Arm()
    ready.set()

    moved = False
    start = arm.elapsed_time()

    while not stop_event.is_set() and arm.myArm.status:
        # if ballXYZ_queue.empty():
        #     # Does nothing if there is no ballXYZ to process
        #     continue

        # ballXYZ = ballXYZ_queue.get_nowait()

        try:
            # phi_cmd = arm.ballXYZ_to_phi_cmd(ballXYZ)
            phi_cmd = np.array([0.2, 0, 0, 0], dtype=np.float64)
            if not moved:
                inp = input("Input commands: ")
                if inp == "home":
                    raise ValueError("Arm homing command received")
                elif len(inp.split(",")) == 4:
                    inp_ls = inp.split(",")
                    phi_cmd = np.array(
                        [
                            float(inp_ls[0]),
                            float(inp_ls[1]),
                            float(inp_ls[2]),
                            float(inp_ls[3]),
                        ],
                        dtype=np.float64,
                    )
                elif inp == "phi":
                    print(arm.phi)
                    continue
                elif inp == "_phi":
                    print(arm._phi)
                    continue
                elif inp == "_phi_offset":
                    print(arm._phi_offset)
                    continue

                print(f"Moving arm with command: {phi_cmd}")
                arm.move(phi_Cmd=phi_cmd)
                moved = True
        except ValueError as e:
            print(f"Command error: {e}")
            arm.home()

        # Only resets moved flag after the sample time has elapsed and we have already
        # moved this sample to ensure that commands are only sent once per sampletime
        if (arm.elapsed_time() - start) > arm.sampleTime and moved:
            moved = False
            start = start + arm.sampleTime

    # terminate only after loop ends and no more read_write_std calls can happen
    try:
        arm.myArm.terminate()
    except Exception as e:
        print(f"Terminate error: {e}")


def main() -> None:
    stop_event = threading.Event()
    ballXYZ_queue: queue.Queue = queue.Queue(maxsize=1000)

    cam_ready = threading.Event()
    arm_ready = threading.Event()

    cam_thread = threading.Thread(target=camera_worker, args=(ballXYZ_queue, stop_event, cam_ready), daemon=False)
    arm_thread = threading.Thread(target=arm_worker, args=(ballXYZ_queue, stop_event, arm_ready), daemon=False)

    cam_thread.start()
    arm_thread.start()

    # # Wait for both to initialize (prevents “use before init” issues)
    # cam_ready.wait()
    # arm_ready.wait()

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
