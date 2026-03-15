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
    phi_cmd: Optional[np.ndarray] = None
    future_robot_points: Optional[np.ndarray] = None  # shape (N, 3)
    last_ballXYZ: Optional[np.ndarray] = None
    last_timestamp: Optional[float] = None


class SharedLatest:
    """Thread-safe latest-value container.

    Stores only the newest sample, which is what you want in a real-time
    camera/control/display pipeline.
    """

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
    """A tiny queue that always keeps the newest item only."""

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
    """Handles either Camera.T_RobotBase_to_Camera(x, y, z) or (..., XYZR)."""
    xyz_robot = np.asarray(xyz_robot, dtype=np.float64).reshape(3)
    try:
        return cam.T_RobotBase_to_Camera(xyz_robot)
    except TypeError:
        return cam.T_RobotBase_to_Camera(xyz_robot[0], xyz_robot[1], xyz_robot[2])


def draw_camera_overlay(frame: np.ndarray, snap: CameraSnapshot) -> np.ndarray:
    out = frame.copy()

    if snap.ball_found and snap.u is not None and snap.v is not None:
        cv2.circle(out, (int(snap.u), int(snap.v)), 5, (255, 0, 0), -1)

    y = 30
    if snap.ball_found and snap.ballXYZ is not None:
        x, yy, z = np.asarray(snap.ballXYZ).reshape(3)
        cv2.putText(
            out,
            f"ballXYZ: [{x:.3f}, {yy:.3f}, {z:.3f}]",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if snap.u is not None and snap.v is not None and snap.z is not None:
        cv2.putText(
            out,
            f"u,v,z,score: ({snap.u}, {snap.v}, {snap.z:.3f}, {0.0 if snap.score is None else snap.score:.3f})",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    if snap.score_parts is not None and len(snap.score_parts) >= 6:
        sp = snap.score_parts
        cv2.putText(
            out,
            f"circ {sp[2]:.3f}, solid {sp[3]:.3f}, aspect {sp[4]:.3f}, color {sp[5]:.3f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def draw_arm_overlay(
    frame: np.ndarray,
    cam: Camera,
    snap: Optional[CameraSnapshot],
    arm_state: Optional[ArmOverlayState],
) -> np.ndarray:
    out = frame.copy()

    if snap is not None and snap.ball_found and snap.ballXYZ is not None:
        x, y, z = np.asarray(snap.ballXYZ).reshape(3)
        cv2.putText(
            out,
            f"ballXYZ: [{x:.3f}, {y:.3f}, {z:.3f}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if arm_state is not None and arm_state.phi_cmd is not None:
        phi = np.asarray(arm_state.phi_cmd).reshape(-1)
        cv2.putText(
            out,
            f"phi_cmd: [{phi[0]:.3f}, {phi[1]:.3f}, {phi[2]:.3f}, {phi[3]:.3f}]",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if arm_state is not None and arm_state.future_robot_points is not None:
        for xyz in np.asarray(arm_state.future_robot_points):
            try:
                u, v = project_robot_point_to_camera(cam, xyz)
                cv2.drawMarker(
                    out,
                    (int(u), int(v)),
                    (255, 255, 255),
                    cv2.MARKER_STAR,
                    10,
                    2,
                )
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
    ready.set()

    while not stop_event.is_set():
        try:
            ballXYZ, ball_found, timestamp = cam.capture_and_process()
            frame = np.asarray(cam.current_frame).copy()

            snapshot = CameraSnapshot(
                frame=frame,
                ballXYZ=None if ballXYZ is None else np.asarray(ballXYZ, dtype=np.float64).reshape(3),
                ball_found=bool(ball_found),
                timestamp=float(timestamp),
                u=cam.u,
                v=cam.v,
                z=cam.z,
                score=cam.score,
                score_parts=None if cam.score_parts is None else tuple(cam.score_parts),
            )
            latest_cam_snapshot.set(snapshot)

            if ballXYZ is not None:
                ballXYZ_queue.put_latest(
                    (
                        np.asarray(ballXYZ, dtype=np.float64).reshape(3),
                        bool(ball_found),
                        float(timestamp),
                    )
                )
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
    arm = Arm()
    ready.set()

    moved = False
    start = arm.elapsed_time()
    future_points_drawn = 5
    timestep = 0.25

    try:
        while not stop_event.is_set() and arm.myArm.status:
            try:
                ballXYZ, ball_found, timestamp = ballXYZ_queue.get(timeout=0.02)
            except queue.Empty:
                continue

            try:
                phi_cmd = arm.ballXYZ_to_phi_cmd(ballXYZ, ball_found, timestamp)

                future_pts = []
                for i in range(future_points_drawn):
                    t_future = timestamp + i * timestep
                    pred = np.asarray(arm.traj.predict_pos(t_future), dtype=np.float64).reshape(3)
                    future_pts.append(pred)
                future_pts_arr = np.asarray(future_pts, dtype=np.float64)

                latest_arm_state.set(
                    ArmOverlayState(
                        phi_cmd=np.asarray(phi_cmd, dtype=np.float64).reshape(4),
                        future_robot_points=future_pts_arr,
                        last_ballXYZ=np.asarray(ballXYZ, dtype=np.float64).reshape(3),
                        last_timestamp=float(timestamp),
                    )
                )

                if not moved:
                    arm.move(phi_Cmd=phi_cmd)
                    moved = True

            except ValueError as e:
                print(f"Command error: {e}")
                arm.home()
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
            snap = latest_cam_snapshot.get()
            arm_state = latest_arm_state.get()

            if snap is not None:
                camera_view = draw_camera_overlay(snap.frame, snap)
                arm_view = draw_arm_overlay(snap.frame, cam, snap, arm_state)

                cv2.imshow("camera_pov", camera_view)
                cv2.imshow("arm_pov", arm_view)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
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