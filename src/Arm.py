# Special QArm library imports
from typing import Optional

from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities

# Standard library imports
import time
import numpy as np

from kalman_ballistic import BallisticKalmanFilter, KalmanInterceptResult


def estimate_time_to_move_xy_ms(current_xy: np.ndarray, target_xy: np.ndarray, v_xy_max_mps: float = 1.0) -> float:
    current_xy = np.asarray(current_xy, dtype=np.float64).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    d = np.linalg.norm(target_xy - current_xy)
    return 1000.0 * float(d / max(v_xy_max_mps, 1e-6))


class Arm:
    def __init__(self) -> None:
        self.startTime = time.time()
        self.sampleRate = 200
        self.sampleTime = 1 / self.sampleRate
        self.myArm = QArm(hardware=1)
        self.myArmUtilities = QArmUtilities()

        self.prev_meas_phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self.prev_meas_pos = np.array([0, 0, 0], dtype=np.float64)

        self.phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64)
        self.pos_cmd = np.array([0, 0, 0], dtype=np.float64)

        self._phi_dot = np.array([0, 0, 0, 0], dtype=np.float64)
        self._R = np.identity(3, dtype=np.float64)
        self._gripper = np.array(0.0, dtype=np.float64)
        self._led = np.array([1, 0, 1], dtype=np.float64)

        self.prev_phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64)
        self.interception_point_ROBOT = None
        self.interception_time = None
        self.last_intercept_valid = False
        self.last_intercept_reason = "startup"

        self.kf = BallisticKalmanFilter()
        self.missed_frames = 0
        self.missed_frames_max = 80

        self.catch_z = 0.26
        self.min_lead_time_ms = 10.0
        self.max_prediction_horizon_ms = 1800.0
        self.v_xy_arm_max_mps = 2.0
        self.reachability_slack = 1.8

        self.T04 = np.identity(4, dtype=np.float64)
        self.L_6 = 0.25

        self.home()

        vel_tol = 0.02
        settle_timeout_s = 5.0
        settle_start = time.time()
        while time.time() - settle_start < settle_timeout_s:
            self.myArm.read_std()
            joint_speed = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
            if np.linalg.norm(joint_speed) <= vel_tol:
                break
            time.sleep(0.05)

        self._phi_offset = np.array(self.myArm.measJointPosition[0:4], dtype=np.float64, copy=True)

    def elapsed_time(self) -> float:
        return time.time() - self.startTime

    def _clear_intercept_only(self) -> None:
        self.interception_point_ROBOT = None
        self.interception_time = None
        self.last_intercept_valid = False

    def _reset_tracking(self, reason: str = "") -> None:
        if reason:
            print(f"Resetting Kalman tracker: {reason}")
        self.kf.reset()
        self._clear_intercept_only()
        self.last_intercept_reason = reason if reason else "reset"

    def _predict_intercept(self) -> KalmanInterceptResult:
        result = self.kf.predict_plane_intercept(
            z_catch=self.catch_z,
            min_lead_ms=self.min_lead_time_ms,
            max_horizon_ms=self.max_prediction_horizon_ms,
        )
        if not result.valid or result.xyz_hit is None:
            return result

        try:
            current_xy = np.asarray(self.pos[:2], dtype=np.float64).reshape(2)
        except Exception:
            current_xy = np.array([0.0, 0.0], dtype=np.float64)

        move_time_ms = estimate_time_to_move_xy_ms(
            current_xy=current_xy,
            target_xy=result.xyz_hit[:2],
            v_xy_max_mps=self.v_xy_arm_max_mps,
        )
        available_time_ms = float(result.t_hit_ms - self.kf.last_t_ms)

        if move_time_ms > self.reachability_slack * max(available_time_ms, 1.0):
            return KalmanInterceptResult(
                valid=False,
                t_hit_ms=result.t_hit_ms,
                xyz_hit=result.xyz_hit,
                xy_hit=result.xy_hit,
                confidence=result.confidence,
                reason="arm cannot reach in time",
            )

        return result

    def ballXYZ_to_phi_cmd(self, XYZ: np.ndarray, ball_found: bool, timestamp: float) -> Optional[np.ndarray]:
        self.last_intercept_valid = False

        if not ball_found or ball_found is None:
            self.missed_frames += 1
            self._clear_intercept_only()
            self.last_intercept_reason = f"tracking missed ({self.missed_frames}/{self.missed_frames_max})"
            if self.missed_frames >= self.missed_frames_max:
                self._reset_tracking(reason="lost tracking")
            return None

        self.missed_frames = 0

        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            self._clear_intercept_only()
            self.last_intercept_reason = "invalid XYZ input"
            return None

        self.kf.update(float(timestamp), xyz_meas)
        result = self._predict_intercept()

        if not result.valid or result.xyz_hit is None:
            self._clear_intercept_only()
            self.last_intercept_reason = result.reason
            print(f"No valid intercept: {result.reason}")
            return None

        ik_xyz = np.asarray(result.xyz_hit, dtype=np.float64).reshape(3)
        self.interception_point_ROBOT = ik_xyz
        self.interception_time = float(result.t_hit_ms)

        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(ik_xyz, 0.0, self.phi)

        phi_seed = np.asarray(self.phi, dtype=np.float64)
        all_solns = np.asarray(ik_all_solns, dtype=np.float64)
        chosen_phi = None

        if all_solns.ndim == 2 and all_solns.shape[0] == 4:
            valid_cols = np.all(np.isfinite(all_solns), axis=0)
            if np.any(valid_cols):
                valid_solns = all_solns[:, valid_cols].T
                idx = np.argmin(np.linalg.norm(valid_solns - phi_seed, axis=1))
                chosen_phi = valid_solns[idx]

        if chosen_phi is None:
            fallback_phi = np.asarray(ik_soln, dtype=np.float64).reshape(-1)
            if fallback_phi.shape == (4,) and np.all(np.isfinite(fallback_phi)):
                chosen_phi = fallback_phi

        if chosen_phi is None:
            self._clear_intercept_only()
            self.last_intercept_reason = "IK failed"
            print("IK failed for predicted intercept; no fresh command.")
            return None

        phi_cmd = np.asarray(chosen_phi, dtype=np.float64)
        self.prev_phi_cmd = phi_cmd
        self.last_intercept_valid = True
        self.last_intercept_reason = "ok"

        print(
            f"Intercept OK: xyz={ik_xyz}, t_hit_ms={self.interception_time:.1f}, "
            f"phi_cmd={phi_cmd}, conf={result.confidence:.3f}"
        )
        return phi_cmd

    def get_future_points(self, n_points: int = 30, step_ms: float = 25.0) -> np.ndarray:
        return self.kf.predict_future_points(n_points=n_points, step_ms=step_ms)

    def get_past_points(self, n_points: int = 20) -> np.ndarray:
        return self.kf.measured_points(n_points=n_points)

    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None) -> bool:
        input_phi_cmd = np.asarray(phi_Cmd, dtype=np.float64)
        if input_phi_cmd.shape != (4,):
            raise ValueError("phi_Cmd must be an iterable of 4 joint angles [rad].")

        r = 0.05

        if np.equal(self.phi_cmd, input_phi_cmd).all():
            return False
        elif np.linalg.norm(input_phi_cmd - self.phi_cmd) <= r:
            return False
        else:
            self.phi_cmd = input_phi_cmd

        if gripper_Cmd is not None:
            gripper_cmd = float(gripper_Cmd)
            if not (0.0 <= gripper_cmd <= 1.0):
                raise ValueError("gripper_Cmd must be between 0 and 1.")
            self._gripper = np.asarray(gripper_cmd, dtype=np.float64)

        if led_Cmd is not None:
            led_cmd = np.asarray(led_Cmd, dtype=np.float64)
            if led_cmd.shape != (3,):
                raise ValueError("led_Cmd must be an iterable of 3 values [R, G, B].")
            if np.any((led_cmd < 0) | (led_cmd > 1)):
                raise ValueError("Each led_Cmd value must be between 0 and 1.")
            self._led = led_cmd

        try:
            self.limit_check(self.phi_cmd)
            self.workspace_check(self.phi_cmd)
        except ValueError as e:
            print(f"Error occurred: {e}")
            return False

        self.myArm.read_write_std(phiCMD=self.phi_cmd, grpCMD=self._gripper, baseLED=self._led)
        print("commanded to phi:", self.phi_cmd, "pos:", self.pos_cmd)
        return True

    def limit_check(self, phi_cmd) -> None:
        if phi_cmd[0] < -np.radians(170) or phi_cmd[0] > np.radians(170):
            raise ValueError("Base Phi limit reached.")
        elif phi_cmd[1] < -np.radians(85) or phi_cmd[1] > np.radians(85):
            raise ValueError("Shoulder Phi limit reached.")
        elif phi_cmd[2] < -np.radians(95) or phi_cmd[2] > np.radians(75):
            raise ValueError("Elbow Phi limit reached.")
        elif phi_cmd[3] < -np.radians(160) or phi_cmd[3] > np.radians(160):
            raise ValueError("Wrist Phi limit reached.")

    def workspace_check(self, phi_cmd) -> None:
        self.pos_cmd, _ = self.qarm_forward_kinematics(phi_cmd)
        if self.pos_cmd[2] < 0.1:
            raise ValueError("End-effector z position limit reached.")

    def home(self) -> None:
        self.phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64)
        self._led = np.array([1, 0, 0], dtype=np.float64)
        self.myArm.read_write_std(phiCMD=self.phi_cmd, grpCMD=self._gripper, baseLED=self._led)

    def qarm_forward_kinematics(self, phi):
        theta = phi.copy()
        theta[0] = phi[0]
        theta[1] = phi[1] + self.myArmUtilities.BETA - np.pi / 2
        theta[2] = phi[2] - self.myArmUtilities.BETA
        theta[3] = phi[3]

        T01 = self.myArmUtilities.quanser_arm_DH(0, -np.pi / 2, self.myArmUtilities.LAMBDA_1, theta[0])
        T12 = self.myArmUtilities.quanser_arm_DH(self.myArmUtilities.LAMBDA_2, 0, 0, theta[1])
        T23 = self.myArmUtilities.quanser_arm_DH(0, -np.pi / 2, 0, theta[2])
        T34 = self.myArmUtilities.quanser_arm_DH(0, 0, self.myArmUtilities.LAMBDA_3 + self.L_6, theta[3])

        T02 = T01 @ T12
        T03 = T02 @ T23
        self.T04 = T03 @ T34

        p4 = self.T04[0:3, 3]
        R04 = self.T04[0:3, 0:3]
        return p4, R04

    def qarm_inverse_kinematics(self, p, gamma, phi_prev):
        theta = np.zeros((4, 4), dtype=np.float64)
        phi = np.zeros((4, 4), dtype=np.float64)

        def inv_kin_setup(p):
            A = self.myArmUtilities.LAMBDA_2
            C = -(self.myArmUtilities.LAMBDA_3 + self.L_6)
            H = self.myArmUtilities.LAMBDA_1 - p[2]
            D1 = -np.sqrt(p[0] ** 2 + p[1] ** 2)
            D2 = np.sqrt(p[0] ** 2 + p[1] ** 2)
            F = (D1**2 + H**2 - A**2 - C**2) / (2 * A)
            return A, C, H, D1, D2, F

        def solve_case_C_j2(j3, A, C, D, H):
            M = A + C * np.sin(j3)
            N = -C * np.cos(j3)
            cos_term = (D * M + H * N) / (M**2 + N**2)
            sin_term = (H - N * cos_term) / M
            j2 = np.arctan2(sin_term, cos_term)
            return j2

        A, C, H, D1, D2, F = inv_kin_setup(p)

        root_term = np.sqrt(np.maximum(C**2 - F**2, 0.0))
        theta[2, 0] = 2 * np.arctan2(C + root_term, F)
        theta[2, 1] = 2 * np.arctan2(C - root_term, F)
        theta[2, 2] = 2 * np.arctan2(C + root_term, F)
        theta[2, 3] = 2 * np.arctan2(C - root_term, F)

        theta[1, 0] = solve_case_C_j2(theta[2, 0], A, C, D1, H)
        theta[1, 1] = solve_case_C_j2(theta[2, 1], A, C, D1, H)
        theta[1, 2] = solve_case_C_j2(theta[2, 2], A, C, D2, H)
        theta[1, 3] = solve_case_C_j2(theta[2, 3], A, C, D2, H)

        theta[0, 0] = np.arctan2(
            p[1] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 0]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 0] + theta[2, 0])),
            p[0] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 0]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 0] + theta[2, 0]))
        )
        theta[0, 1] = np.arctan2(
            p[1] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 1]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 1] + theta[2, 1])),
            p[0] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 1]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 1] + theta[2, 1]))
        )
        theta[0, 2] = np.arctan2(
            p[1] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 2]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 2] + theta[2, 2])),
            p[0] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 2]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 2] + theta[2, 2]))
        )
        theta[0, 3] = np.arctan2(
            p[1] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 3]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 3] + theta[2, 3])),
            p[0] / (self.myArmUtilities.LAMBDA_2 * np.cos(theta[1, 3]) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin(theta[1, 3] + theta[2, 3]))
        )

        phi[0, :] = theta[0, :]
        phi[1, :] = theta[1, :] - self.myArmUtilities.BETA + np.pi / 2
        phi[2, :] = theta[2, :] + self.myArmUtilities.BETA
        phi[3, :] = gamma * np.ones((4))

        phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi

        phiOptimal = phi[:, 0]
        if np.linalg.norm(phiOptimal - phi_prev) > np.linalg.norm(phi[:, 1] - phi_prev):
            phiOptimal = phi[:, 1]
        if np.linalg.norm(phiOptimal - phi_prev) > np.linalg.norm(phi[:, 2] - phi_prev):
            phiOptimal = phi[:, 2]
        if np.linalg.norm(phiOptimal - phi_prev) > np.linalg.norm(phi[:, 3] - phi_prev):
            phiOptimal = phi[:, 3]

        return phi, phiOptimal

    @property
    def phi(self):
        self.myArm.read_std()
        self.prev_meas_phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64) - self._phi_offset
        return self.prev_meas_phi

    @property
    def pos(self) -> np.ndarray:
        self.prev_meas_pos, self._R = self.qarm_forward_kinematics(self.phi)
        return self.prev_meas_pos

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def gripper(self) -> np.ndarray:
        return self._gripper

    @property
    def led(self) -> np.ndarray:
        return self._led

    @property
    def phi_dot(self) -> np.ndarray:
        self._phi_dot = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        return self._phi_dot