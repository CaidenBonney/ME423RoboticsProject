import cv2
import numpy as np
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from Arm import Arm
from Camera import Camera


@dataclass
class CameraSnapshot:
    """ Data class for camera related information. Camera side of the overlay. """
    frame: np.ndarray
    ballXYZ: Optional[np.ndarray]
    ball_found: bool
    timestamp: float
    u: Optional[int] = None
    v: Optional[int] = None
    z: Optional[float] = None
    score: Optional[float] = None
    score_parts: Optional[object] = None


@dataclass
class ArmOverlayState:
    """ Data class for arm related information. Arm side of the overlay. """
    phi_cmd: Optional[np.ndarray] = None
    pos_cmd: Optional[np.ndarray] = None
    future_robot_points: Optional[np.ndarray] = None
    past_robot_points: Optional[np.ndarray] = None
    last_ballXYZ: Optional[np.ndarray] = None
    last_timestamp: Optional[float] = None
    interception_point_ROBOT: Optional[np.ndarray] = None
    interception_time: Optional[float] = None
    trajMade: bool = False


class SharedLatest:
    """ SharedLatest is a thread-safe container for a single value. """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


class LatestQueue:
    """ LatestQueue is a thread-safe queue for a single value. """
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=1)

    def put_latest(self, item) -> None:
        try:
            if self._q.full():
                self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass

    def get(self, timeout: float):
        return self._q.get(timeout=timeout)


def project_robot_point_to_camera(cam: Camera, xyz_robot: np.ndarray) -> tuple[int, int]:
    """Formats and executes base-to-camera position transformation."""
    xyz_robot = np.asarray(xyz_robot, dtype=np.float64).reshape(3)
    try:
        return cam.T_RobotBase_to_Camera(xyz_robot)
    except TypeError:
        return cam.T_RobotBase_to_Camera(xyz_robot[0], xyz_robot[1], xyz_robot[2])


def draw_camera_overlay(frame: np.ndarray, snap: CameraSnapshot) -> np.ndarray:
    """Draws the camera overlay on the frame. Camera side information."""
    out = frame.copy()

    # Draw the ball if it was found
    if snap.ball_found and snap.u is not None and snap.v is not None:
        cv2.circle(out, (int(snap.u), int(snap.v)), 5, (255, 0, 0), -1)

    y = 30
    # Print the ballXYZ if it was found
    if snap.ball_found and snap.ballXYZ is not None:
        x, yy, z = np.asarray(snap.ballXYZ).reshape(3)
        cv2.putText(out, f"ballXYZ: [{x:.3f}, {yy:.3f}, {z:.3f}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 30

    # Print the ball position [pix], depth reading, and score if found
    if snap.u is not None and snap.v is not None and snap.z is not None:
        cv2.putText(out,
                    f"u,v,z,score: ({snap.u}, {snap.v}, {snap.z:.3f}, {0.0 if snap.score is None else snap.score:.3f})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    # Print the score parts if found
    if snap.score_parts is not None and len(snap.score_parts) >= 6:
        sp = snap.score_parts
        cv2.putText(out,
                    f"circ {sp[2]:.3f}, solid {sp[3]:.3f}, aspect {sp[4]:.3f}, color {sp[5]:.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def draw_arm_overlay(
    frame: np.ndarray,
    cam: Camera,
    snap: Optional[CameraSnapshot],
    arm_state: Optional[ArmOverlayState],
) -> np.ndarray:
    """Draws the arm overlay on the frame. Arm side information."""

    out = frame.copy()
    y_offset = 30
    # Print the ballXYZ if it was found
    if snap is not None and snap.ball_found and snap.ballXYZ is not None:
        x, y, z = np.asarray(snap.ballXYZ).reshape(3)
        cv2.putText(out, f"ballXYZ: [{x:.3f}, {y:.3f}, {z:.3f}]",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30

    # Print the phi_cmd if it was found
    if arm_state is not None and arm_state.phi_cmd is not None:
        phi = np.asarray(arm_state.phi_cmd).reshape(-1)
        cv2.putText(out,
                    f"phi_cmd: [{phi[0]:.3f}, {phi[1]:.3f}, {phi[2]:.3f}, {phi[3]:.3f}]",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30

    # Print the pos_cmd (interception point) if it was found
    if arm_state is not None and arm_state.pos_cmd is not None:
        pos = np.asarray(arm_state.pos_cmd).reshape(-1)
        cv2.putText(out, f"pos_cmd: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30

    # Print the timestamp if it was found
    if snap is not None and snap.timestamp is not None:
        cv2.putText(out, f"timestamp: {snap.timestamp}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30

    # Print if a trajectory object is made (turns false once reset is called) 
    if snap is not None and arm_state is not None:
        cv2.putText(out, f"Trajectoy made: {arm_state.trajMade}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30

    # Draws the future projected points by the trajectory object
    if arm_state is not None and arm_state.future_robot_points is not None:
        for xyz in np.asarray(arm_state.future_robot_points):
            try:
                u, v = project_robot_point_to_camera(cam, xyz)
                cv2.drawMarker(out, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
            except Exception:
                pass

    # Draws the interception point
    if arm_state is not None:
        interception_xyz = arm_state.interception_point_ROBOT
        if interception_xyz is not None:
            try:
                u, v = project_robot_point_to_camera(cam, interception_xyz)
                cv2.drawMarker(out, (int(u), int(v)), (255, 255, 255), cv2.MARKER_DIAMOND, 30, 3)
            except Exception:
                pass

    # Draws the past detected positions
    if arm_state is not None and arm_state.past_robot_points is not None:
        for xyz in np.asarray(arm_state.past_robot_points):
            try:
                u, v = project_robot_point_to_camera(cam, xyz)
                cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_STAR, 10, 2)
            except Exception:
                pass
    return out


def camera_worker(
    cam: Camera,
    latest_cam_snapshot: SharedLatest,
    ballXYZ_queue: LatestQueue,
    stop_event: threading.Event,
    ready: threading.Event,
) -> None:
    """Camera worker thread. Captures images, finds the ball [pix], finds the 3D position w.r.t. the base frame and pushes them to the queue."""

    ready.set()
    while not stop_event.is_set():
        try:
            ballXYZ, ball_found, timestamp = cam.capture_and_process()
            
            # Create and push the camera side information to the threadsafe objects
            snapshot = CameraSnapshot(
                frame=cam.current_frame.copy() if cam.current_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                ballXYZ=np.asarray(ballXYZ, dtype=np.float64).reshape(3) if ballXYZ is not None else None,
                ball_found=bool(ball_found),
                timestamp=float(timestamp),
                u=cam.u,
                v=cam.v,
                z=cam.z,
                score=cam.score,
                score_parts=None if cam.score_parts is None else tuple(cam.score_parts),
            )
            latest_cam_snapshot.set(snapshot)

            # Push the ballXYZ and ball_found as long as a frame was captured
            if timestamp is not None:
                ballXYZ_queue.put_latest((
                    np.asarray(ballXYZ, dtype=np.float64).reshape(3) if ballXYZ is not None else None,
                    bool(ball_found),
                    float(timestamp),
                ))
        except Exception as e:
            print(f"camera_worker error: {e}")
            time.sleep(0.01)


def arm_worker(
    latest_cam_snapshot: SharedLatest,
    latest_arm_state: SharedLatest,
    ballXYZ_queue: LatestQueue,
    stop_event: threading.Event,
    ready: threading.Event,
) -> None:
    """Arm worker thread. 
    - Gathers XYZ information stream
    - Runs the trajectory object and predicts the future points
    - Finds interception with desired z-plane
    - Pushes the arm side information to the queue.
    """

    # Creating the arm object (which communicates with the hardware) is important to happen in its own thread. 
    # Hardware communication is not thread agnostic.
    arm = Arm() 
    ready.set()

    moved = False
    start = arm.elapsed_time()
    future_points_drawn = 20 # Number of future points to draw
    past_points_drawn = 20 # Number of past points to draw
    timestep = 20 # Time step [ms] for the trajectory object

    try:
        while not stop_event.is_set() and arm.myArm.status:
            try:
                # Grab from the queue
                ballXYZ, ball_found, timestamp = ballXYZ_queue.get(timeout=0.02)
            except queue.Empty:
                print("Ball XYZ queue is empty")
                continue

            try:
                # performs the XYZ-to-phi command. Trajectory object used for interception calculations.
                phi_cmd = arm.ballXYZ_to_phi_cmd_ballistic(ballXYZ, ball_found, timestamp)
                interception_point_ROBOT = arm.interception_point_ROBOT
                interception_time = arm.interception_time

                # Calculates the future points for the trajectory object
                future_pts = []
                if arm.ballistic_interceptor._valid:
                    for i in range(future_points_drawn):
                        t_future = arm.ballistic_interceptor._t[-1] + i * timestep
                        pred = np.asarray(arm.ballistic_interceptor.predict_pos(t_future), dtype=np.float64).reshape(3)
                        future_pts.append(pred)
                future_pts_arr = np.asarray(future_pts, dtype=np.float64)

                # Grabs past points from the trajectory object
                past_pts = np.asarray(arm.ballistic_interceptor._pos[-past_points_drawn:, :], dtype=np.float64)

                # Push the arm side information to the queue
                latest_arm_state.set(
                    ArmOverlayState(
                        phi_cmd=np.asarray(arm.phi_cmd, dtype=np.float64).reshape(4),
                        pos_cmd=np.asarray(arm.pos_cmd, dtype=np.float64).reshape(3),
                        future_robot_points=future_pts_arr,
                        past_robot_points=past_pts,
                        last_ballXYZ=np.asarray(ballXYZ, dtype=np.float64).reshape(3) if ballXYZ is not None else None,
                        last_timestamp=float(timestamp),
                        interception_point_ROBOT=(
                            np.asarray(interception_point_ROBOT, dtype=np.float64).reshape(3)
                            if interception_point_ROBOT is not None else None
                        ),
                        interception_time=float(interception_time) if interception_time is not None else None,
                        trajMade = arm.ballistic_interceptor._valid
                    )
                )

                # Moves the arm to the phi_cmd
                if not moved:
                    arm.move(phi_Cmd=phi_cmd)
                    moved = True


            except ValueError as e:
                print(f"Command error: {e}")
            except EOFError:
                break
            except Exception as e:
                print(f"arm_worker loop error: {e}")

            # Sampletime limits the rate at which the arm moves 
            if (arm.elapsed_time() - start) > arm.sampleTime and moved:
                moved = False
                start = start + arm.sampleTime
    finally:
        try:
            arm.myArm.terminate()
        except Exception as e:
            print(f"Terminate error: {e}")
        stop_event.set()


def manual_control_arm_worker(
    latest_cam_snapshot: SharedLatest,
    latest_arm_state: SharedLatest,
    ballXYZ_queue: LatestQueue,
    stop_event: threading.Event,
    ready: threading.Event,
) -> None:
    """Thread for manual control of the arm.
    """

    arm = Arm()
    ready.set()

    moved = False
    start = arm.elapsed_time()
    test_pose_1 = np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    test_pose_2 = np.array([0.35, -0.2, 0.15, 0.0], dtype=np.float64)
    try:
        phi_cmd = test_pose_1.copy()
        while not stop_event.is_set() and arm.myArm.status:
            try:
                if not moved:
                    inp = input("test input: ")
                    if inp == "home":
                        arm.home()
                        time.sleep(1.5)
                        arm.print_measurement_check("after moving home")
                        continue
                    elif inp == "test1":
                        phi_cmd = test_pose_1.copy()
                    elif inp == "test2":
                        phi_cmd = test_pose_2.copy()
                    elif inp == "check":
                        arm.print_measurement_check("manual measurement check")
                        continue
                    elif len(inp.split(",")) == 4:
                        inp_ls = inp.split(",")
                        phi_cmd = np.array([float(x) for x in inp_ls], dtype=np.float64)
                    elif inp == "phi":
                        print(arm.phi)
                        continue
                    elif inp == "_phi":
                        print(arm.prev_meas_phi)
                        continue
                    elif inp == "_phi_offset":
                        print(arm._phi_offset)
                        continue

                    print(f"moving arm with command: {phi_cmd}")
                    arm.move(phi_Cmd=phi_cmd)
                    time.sleep(1.5)
                    arm.print_measurement_check(f"after moving to {phi_cmd}")
                    moved = True

            except ValueError as e:
                print(f"Command error: {e}, phi_cmd: {phi_cmd}")
                arm.home()
            except EOFError:
                break
            except Exception as e:
                print(f"arm_worker loop error: {e}")
            if (arm.elapsed_time() - start) > arm.sampleTime and moved:
                moved = False
                start = start + arm.sampleTime
    finally:
        try:
            arm.myArm.terminate()
        except Exception as e:
            print(f"Terminate error: {e}")
        stop_event.set()


def main() -> None:
    stop_event = threading.Event()

    cam = Camera()
    latest_cam_snapshot = SharedLatest()
    latest_arm_state = SharedLatest()
    ballXYZ_queue = LatestQueue()

    cam_ready = threading.Event()
    arm_ready = threading.Event()

    cam_thread = threading.Thread(
        target=camera_worker,
        args=(cam, latest_cam_snapshot, ballXYZ_queue, stop_event, cam_ready),
        daemon=False,
    )

    arm_thread = threading.Thread(
        target=arm_worker,
        args=(latest_cam_snapshot, latest_arm_state, ballXYZ_queue, stop_event, arm_ready),
        daemon=False,
    )

    cam_thread.start()
    arm_thread.start()

    cam_ready.wait()
    arm_ready.wait()

    try:
        while arm_thread.is_alive() and not stop_event.is_set():
            # Grab the latest camera and arm information
            snap = latest_cam_snapshot.get()
            arm_state = latest_arm_state.get()

            # Draw the overlay
            if snap is not None and arm_state is not None:
                arm_view = draw_arm_overlay(snap.frame, cam, snap, arm_state)
                cv2.imshow("arm_pov", arm_view)

            key = cv2.waitKey(1)
            if key == 27:
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        stop_event.set()
        cam_thread.join()
        arm_thread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
