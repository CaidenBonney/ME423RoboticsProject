import time
import cv2
import numpy as np

from Arm import Arm
from Camera import Camera


def project_robot_point_to_camera(cam: Camera, xyz_robot: np.ndarray) -> tuple[int, int]:
    """
    Be tolerant to either Camera.T_RobotBase_to_Camera(xyz)
    or Camera.T_RobotBase_to_Camera(x, y, z).
    """
    xyz_robot = np.asarray(xyz_robot, dtype=np.float64).reshape(3)
    try:
        uv = cam.T_RobotBase_to_Camera(xyz_robot)
    except TypeError:
        uv = cam.T_RobotBase_to_Camera(xyz_robot[0], xyz_robot[1], xyz_robot[2])

    uv = np.asarray(uv).reshape(2)
    return int(round(uv[0])), int(round(uv[1]))


def draw_camera_overlay(cam_frame: np.ndarray, cam: Camera, ballXYZ, ball_found: bool) -> np.ndarray:
    out = cam_frame.copy()

    if cam.u is not None and cam.v is not None:
        cv2.circle(out, (int(cam.u), int(cam.v)), 5, (255, 0, 0), -1)

    y = 30
    if ball_found and ballXYZ is not None:
        x_b, y_b, z_b = np.asarray(ballXYZ, dtype=np.float64).reshape(3)
        cv2.putText(
            out,
            f"ballXYZ: [{x_b:.3f}, {y_b:.3f}, {z_b:.3f}]",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if cam.u is not None and cam.v is not None and cam.z is not None:
        score_val = 0.0 if cam.score is None else float(cam.score)
        cv2.putText(
            out,
            f"u, v, z, score: ({int(cam.u)}, {int(cam.v)}, {float(cam.z):.3f}, {score_val:.3f})",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if cam.score_parts is not None and len(cam.score_parts) >= 6:
        sp = cam.score_parts
        cv2.putText(
            out,
            f"circ {sp[2]:.3f}, solid {sp[3]:.3f}, aspect {sp[4]:.3f}, color {sp[5]:.3f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 25

        cv2.putText(
            out,
            f"circ {sp[2]/350.0:.3f}, solid {sp[3]/250.0:.3f}, aspect {sp[4]/-60.0:.3f}, color {sp[5]/-90.0:.3f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def draw_arm_overlay(
    arm_frame: np.ndarray,
    cam: Camera,
    ballXYZ,
    ball_found: bool,
    phi_cmd,
    traj_points_robot,
) -> np.ndarray:
    out = arm_frame.copy()

    if ball_found and ballXYZ is not None:
        x_b, y_b, z_b = np.asarray(ballXYZ, dtype=np.float64).reshape(3)
        cv2.putText(
            out,
            f"ballXYZ: [{x_b:.3f}, {y_b:.3f}, {z_b:.3f}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if phi_cmd is not None:
        phi = np.asarray(phi_cmd, dtype=np.float64).reshape(4)
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

    if traj_points_robot is not None:
        for xyz in traj_points_robot:
            try:
                u, v = project_robot_point_to_camera(cam, xyz)
                cv2.drawMarker(
                    out,
                    (u, v),
                    (255, 255, 255),
                    cv2.MARKER_STAR,
                    10,
                    2,
                )
            except Exception:
                pass

    return out


def main() -> None:
    cam = Camera()
    arm = Arm()

    moved = False
    last_move_time = arm.elapsed_time()

    # Display / prediction tuning
    future_points_drawn = 5
    future_dt = 0.05  # 50 ms spacing looks much smoother than 0.25 s

    # Keep latest useful state for display even if the current frame misses the ball
    last_ballXYZ = None
    last_ball_found = False
    last_timestamp = None
    last_phi_cmd = None

    try:
        while arm.myArm.status:
            # 1) Capture + process one frame
            result = cam.capture_and_process()
            if result is None:
                key = cv2.waitKey(1)
                if key == 27:
                    break
                continue

            ballXYZ, ball_found, timestamp = result
            frame = cam.current_frame.copy()

            if ball_found and ballXYZ is not None:
                last_ballXYZ = np.asarray(ballXYZ, dtype=np.float64).reshape(3)
                last_ball_found = True
                last_timestamp = float(timestamp)
            else:
                last_ball_found = False

            # 2) Compute arm command in the same loop, with no threading
            phi_cmd = None
            if ballXYZ is not None:
                try:
                    phi_cmd = arm.ballXYZ_to_phi_cmd(ballXYZ, ball_found, timestamp)
                    last_phi_cmd = np.asarray(phi_cmd, dtype=np.float64).reshape(4)

                    # Rate-limit arm writes using the arm's sample time
                    now_arm = arm.elapsed_time()
                    if (now_arm - last_move_time) >= arm.sampleTime:
                        arm.move(phi_Cmd=phi_cmd)
                        moved = True
                        last_move_time = now_arm

                except ValueError as e:
                    print(f"Command error: {e}")
                    try:
                        arm.home()
                    except Exception as home_error:
                        print(f"Home error: {home_error}")
                    moved = False
                except Exception as e:
                    print(f"Control error: {e}")
                    moved = False

            # 3) Build future trajectory markers using CURRENT time
            #    This removes the stale-marker effect from the threaded version.
            traj_points_robot = None
            try:
                if getattr(arm.traj, "t", np.array([])).size > 0:
                    t_now = time.time()
                    t_draw = t_now + np.arange(future_points_drawn, dtype=np.float64) * future_dt
                    pred = arm.traj.predict_pos(t_draw)  # shape expected: (3, N)

                    pred = np.asarray(pred, dtype=np.float64)
                    if pred.ndim == 2 and pred.shape[0] == 3:
                        traj_points_robot = pred.T  # shape (N, 3)
                    elif pred.ndim == 2 and pred.shape[1] == 3:
                        traj_points_robot = pred
                    elif pred.ndim == 1 and pred.size == 3:
                        traj_points_robot = pred.reshape(1, 3)
            except Exception as e:
                print(f"Prediction draw error: {e}")
                traj_points_robot = None

            # 4) Draw both views from the same frame and same loop iteration
            camera_view = draw_camera_overlay(
                cam_frame=frame,
                cam=cam,
                ballXYZ=last_ballXYZ,
                ball_found=last_ball_found,
            )

            arm_view = draw_arm_overlay(
                arm_frame=frame,
                cam=cam,
                ballXYZ=last_ballXYZ,
                ball_found=last_ball_found,
                phi_cmd=last_phi_cmd,
                traj_points_robot=traj_points_robot,
            )

            cv2.imshow("camera_pov", camera_view)
            cv2.imshow("arm_pov", arm_view)

            # 5) UI event pump
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            arm.myArm.terminate()
        except Exception as e:
            print(f"Terminate error: {e}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()