# Special QArm library imports
from typing import Optional

from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities

# Standard library imports
import time
import numpy as np
from Trajectory import Trajectory

# Interceptor imports
from intercept_1_ballistic import BallisticInterceptor
from intercept_2_kalman import KalmanBallTracker
from intercept_3_reachability import ReachabilityInterceptor
from intercept_4_ransac import RansacBallFitter
from intercept_5_ewma import EWMAInterceptor


class Arm:
    def __init__(self) -> None:
        # Internal variables from basic position mode py files
        self.startTime = time.time()
        self.sampleRate = 200
        self.sampleTime = 1 / self.sampleRate
        self.myArm = QArm(hardware=1)
        self.myArmUtilities = QArmUtilities()

        # Internal variables for current state of the arm
        self.prev_meas_phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self.prev_meas_pos = np.array([0, 0, 0], dtype=np.float64)

        self.phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64)
        self.pos_cmd = np.array([0, 0, 0], dtype=np.float64)

        self._phi_dot = np.array([0, 0, 0, 0], dtype=np.float64)
        self._R = np.identity(3, dtype=np.float64)
        self._gripper = np.array(0.0, dtype=np.float64)
        self._led = np.array([1, 0, 1], dtype=np.float64)

        self._pos_q_max = 100
        self._pos_q = np.empty((self._pos_q_max, 3), dtype=np.float64)
        self._time_q = np.empty(self._pos_q_max, dtype=np.float64)
        self._q_write_idx = 0
        self._q_count = 0

        self.traj = Trajectory()
        self.traj_number = 0
        self.missed_frames = 0
        self.missed_frames_max = 20
        self.prev_phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64)
        self.interception_point_ROBOT = None
        self.interception_time = None

        self.T04 = np.identity(4, dtype=np.float64)
        self.L_6 = 0.25

        # ── Interceptor instances (set catch_z to your desired catch plane height) ──
        # Change catch_z here to match your physical setup.
        _catch_z = 0.10   # [m] height of catch plane above robot base

        self.ballistic_interceptor   = BallisticInterceptor(catch_z=_catch_z)
        self.kalman_tracker          = KalmanBallTracker(catch_z=_catch_z)
        self.reachability_interceptor = ReachabilityInterceptor(catch_z=_catch_z)
        self.ransac_fitter           = RansacBallFitter(catch_z=_catch_z)
        self.ewma_interceptor        = EWMAInterceptor(catch_z=_catch_z)

        self.home()

        # Wait for arm to settle at home
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

    # ── helpers shared by all ballXYZ_to_phi_cmd_* methods ────────────────

    def _reset_all_interceptors(self) -> None:
        """Reset every interceptor and the Trajectory on tracking loss."""
        self.traj = Trajectory()
        self.traj_number += 1
        self.ballistic_interceptor.reset()
        self.kalman_tracker.reset()
        self.reachability_interceptor.reset()
        self.ransac_fitter.reset()
        self.ewma_interceptor.reset()

    def _resolve_ik(self, target_xyz: np.ndarray) -> Optional[np.ndarray]:
        """
        Run IK for *target_xyz* and return the joint-angle solution closest to
        the current arm configuration, or None if IK fails.
        """
        phi_seed = np.asarray(self.phi, dtype=np.float64)
        try:
            all_solns, ik_soln = self.qarm_inverse_kinematics(target_xyz, 0, phi_seed)
        except Exception as e:
            print(f"IK exception: {e}")
            return None

        all_solns = np.asarray(all_solns, dtype=np.float64)
        chosen_phi = None

        if all_solns.ndim == 2 and all_solns.shape[0] == 4:
            valid_cols = np.all(np.isfinite(all_solns), axis=0)
            if np.any(valid_cols):
                valid_solns = all_solns[:, valid_cols].T
                idx = np.argmin(np.linalg.norm(valid_solns - phi_seed, axis=1))
                chosen_phi = valid_solns[idx]

        if chosen_phi is None:
            fallback = np.asarray(ik_soln, dtype=np.float64).reshape(-1)
            if fallback.shape == (4,) and np.all(np.isfinite(fallback)):
                chosen_phi = fallback

        return chosen_phi

    def _apply_phi_cmd(self, phi_cmd: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        """Store, move, and return a resolved phi command."""
        phi_cmd = np.asarray(phi_cmd, dtype=np.float64)
        self.phi_cmd = phi_cmd.copy()
        self.pos_cmd = intercept.copy()
        self.prev_phi_cmd = phi_cmd.copy()
        self.move(phi_Cmd=phi_cmd)
        return phi_cmd

    # ── original methods (unchanged) ──────────────────────────────────────

    def elapsed_time(self) -> float:
        return time.time() - self.startTime

    def print_measurement_check(self, label: str = "measurement check") -> None:
        self.myArm.read_std()
        raw_phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)
        joint_speed = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        rel_phi = raw_phi - self._phi_offset
        print(label)
        print("raw measured phi:", raw_phi)
        print("stored home offset:", self._phi_offset)
        print("home-relative phi:", rel_phi)
        print("measured joint speed:", joint_speed)

    def ballXYZ_to_phi_cmd(self, XYZ: np.ndarray, ball_found: bool, timestamp: float) -> Optional[np.ndarray]:
        """Original polynomial-fit method (unchanged)."""
        if not ball_found or ball_found is None:
            self.missed_frames += 1
            if self.missed_frames == self.missed_frames_max:
                print("Creating new trajectory object, lost tracking", self.traj_number, "time: ", timestamp)
                self.traj = Trajectory()
                self.traj_number += 1
            return self.prev_phi_cmd

        self.missed_frames = 0
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        ik_xyz = xyz_meas
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)
        if self.traj.t.size < 3:
            print("Not enough points to fit parabola, using most recent measurement for IK.")
            return self.prev_phi_cmd
        else:
            print("Fitting parabola,-----------------------------------------")

        pz = self.traj.pz.copy()
        pz[-1] -= 0.1
        z_roots = np.roots(pz)

        real_dt = z_roots[np.abs(z_roots.imag) < 1e-8].real

        t_now_shift = timestamp - self.traj.t0
        fut_dt = real_dt[real_dt >= t_now_shift]

        if fut_dt.size == 0:
            self.interception_point_ROBOT = None
            self.interception_time = None
            return self.prev_phi_cmd

        t_hit_ms = self.traj.t0[0] + np.min(fut_dt)
        print(f"Future hit: {t_hit_ms + timestamp}, fut_dt: {fut_dt}")

        ik_xyz = self.traj.predict_pos(t_hit_ms)[:, 0]
        self.interception_point_ROBOT = ik_xyz
        self.interception_time = t_hit_ms

        ik_pos_cmd = ik_xyz

        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(ik_pos_cmd, 0, self.phi)

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
            raise ValueError("IK failed: target appears unreachable.")
        else:
            print("commanded phi: ", chosen_phi, timestamp)

        phi_cmd = np.asarray(chosen_phi, dtype=np.float64)
        self.prev_phi_cmd = phi_cmd
        return phi_cmd

    def ballXYZ_to_phi_cmd_no_traj(self, XYZ: np.ndarray, ball_found: bool, timestamp: float) -> Optional[np.ndarray]:
        """Direct IK on raw measurement, no trajectory (unchanged)."""
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        self.interception_point_ROBOT = xyz_meas
        self.interception_time = timestamp

        ik_pos_cmd = xyz_meas

        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(ik_pos_cmd, 0, self.phi)

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
            raise ValueError("IK failed: target appears unreachable.")
        else:
            print("commanded phi: ", chosen_phi, timestamp)

        phi_cmd = np.asarray(chosen_phi, dtype=np.float64)
        self.prev_phi_cmd = phi_cmd
        return phi_cmd

    def ballXYZ_to_phi_cmd_no_traj_fixed_xz(self, XYZ: np.ndarray, ball_found: bool, timestamp: float, fixed_x: float, fixed_z: float) -> Optional[np.ndarray]:
        """Fixed-XZ plane interception (unchanged)."""
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        xyz_meas[0] = fixed_x
        xyz_meas[2] = fixed_z

        if not ball_found or ball_found is None:
            return self.prev_phi_cmd

        self.interception_point_ROBOT = xyz_meas
        self.interception_time = timestamp

        ik_pos_cmd = xyz_meas

        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(ik_pos_cmd, 0, self.phi)

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
            raise ValueError("IK failed: target appears unreachable.")
        else:
            print("commanded phi: ", chosen_phi, timestamp)

        phi_cmd = np.asarray(chosen_phi, dtype=np.float64)
        self.prev_phi_cmd = phi_cmd
        return phi_cmd

    # ── NEW: Approach 1 — Physics-first ballistic ──────────────────────────

    def ballXYZ_to_phi_cmd_ballistic(
        self,
        XYZ: np.ndarray,
        ball_found: bool,
        timestamp: float,
    ) -> np.ndarray:
        """
        Intercept using physics-first ballistic prediction.

        Fits the trajectory with gravity as a fixed constant (not a free
        parameter), giving a stable closed-form catch-plane intersection
        with only 3+ observations needed.

        Recommended starting point. Set catch_z in __init__ above.
        """
        if not ball_found:
            self.missed_frames += 1
            if self.missed_frames >= self.missed_frames_max:
                print(f"[ballistic] tracking lost, resetting (traj #{self.traj_number})")
                self._reset_all_interceptors()
            return self.prev_phi_cmd

        self.missed_frames = 0
        xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            return self.prev_phi_cmd

        self.ballistic_interceptor.update(timestamp, xyz)
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        intercept = self.ballistic_interceptor.predict_interception(timestamp)
        if intercept is None:
            print("[ballistic] waiting for enough points or no future root")
            self.interception_point_ROBOT = None
            self.interception_time = None
            return self.prev_phi_cmd

        print(f"[ballistic] intercept={np.round(intercept, 3)}")
        self.interception_point_ROBOT = intercept

        phi_cmd = self._resolve_ik(intercept)
        if phi_cmd is None:
            print("[ballistic] IK failed for intercept point")
            return self.prev_phi_cmd

        return self._apply_phi_cmd(phi_cmd, intercept)

    # ── NEW: Approach 2 — Kalman filter ───────────────────────────────────

    def ballXYZ_to_phi_cmd_kalman(
        self,
        XYZ: np.ndarray,
        ball_found: bool,
        timestamp: float,
    ) -> np.ndarray:
        """
        Intercept using a 6-state Kalman filter (x, y, z, vx, vy, vz) with
        gravity in the process model.

        Best choice when camera measurements are noisy. Tune sigma_pos and
        sigma_process in KalmanBallTracker.__init__ to match your sensor.
        """
        if not ball_found:
            self.missed_frames += 1
            if self.missed_frames >= self.missed_frames_max:
                print(f"[kalman] tracking lost, resetting (traj #{self.traj_number})")
                self._reset_all_interceptors()
            return self.prev_phi_cmd

        self.missed_frames = 0
        xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            return self.prev_phi_cmd

        self.kalman_tracker.update(timestamp, xyz)
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        intercept = self.kalman_tracker.predict_interception(timestamp)
        if intercept is None:
            print("[kalman] waiting for convergence or no future root")
            self.interception_point_ROBOT = None
            self.interception_time = None
            return self.prev_phi_cmd

        print(f"[kalman] intercept={np.round(intercept, 3)}")
        self.interception_point_ROBOT = intercept

        phi_cmd = self._resolve_ik(intercept)
        if phi_cmd is None:
            print("[kalman] IK failed for intercept point")
            return self.prev_phi_cmd

        return self._apply_phi_cmd(phi_cmd, intercept)

    # ── NEW: Approach 3 — Time-aware reachability ─────────────────────────

    def ballXYZ_to_phi_cmd_reachability(
        self,
        XYZ: np.ndarray,
        ball_found: bool,
        timestamp: float,
    ) -> np.ndarray:
        """
        Intercept at the earliest point along the ballistic arc that the arm
        can physically reach before the ball arrives.

        Scans the future arc in time increments, runs IK at each sample, and
        estimates arm travel time from current joint angles via joint velocity
        limits. Picks the soonest reachable catch point.

        Use this when you notice the arm being commanded to impossible targets.
        Tune JOINT_VEL_MAX_RAD_MS and ARM_TRAVEL_SAFETY_FACTOR in
        intercept_3_reachability.py to match your hardware.
        """
        if not ball_found:
            self.missed_frames += 1
            if self.missed_frames >= self.missed_frames_max:
                print(f"[reachability] tracking lost, resetting (traj #{self.traj_number})")
                self._reset_all_interceptors()
            return self.prev_phi_cmd

        self.missed_frames = 0
        xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            return self.prev_phi_cmd

        self.reachability_interceptor.update(timestamp, xyz)
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        # Pass self so the interceptor can call self.qarm_inverse_kinematics
        # and self.phi without needing its own copy of the arm model.
        intercept, phi_cmd = self.reachability_interceptor.predict_reachable_intercept(
            timestamp, self
        )

        if intercept is None or phi_cmd is None:
            print("[reachability] no reachable interception in scan window")
            self.interception_point_ROBOT = None
            self.interception_time = None
            return self.prev_phi_cmd

        print(f"[reachability] intercept={np.round(intercept, 3)}")
        self.interception_point_ROBOT = intercept

        return self._apply_phi_cmd(phi_cmd, intercept)

    # ── NEW: Approach 4 — RANSAC robust fitting ───────────────────────────

    def ballXYZ_to_phi_cmd_ransac(
        self,
        XYZ: np.ndarray,
        ball_found: bool,
        timestamp: float,
    ) -> np.ndarray:
        """
        Intercept using RANSAC-robust ballistic fitting.

        Randomly samples subsets of observations, fits the physics model to
        each, and keeps the fit with the most inliers. Effectively discards
        spurious detections before fitting.

        Use this when your camera occasionally produces outlier detections
        (reflections, partial occlusions, misidentified objects).
        Tune inlier_threshold in RansacBallFitter.__init__.
        """
        if not ball_found:
            self.missed_frames += 1
            if self.missed_frames >= self.missed_frames_max:
                print(f"[ransac] tracking lost, resetting (traj #{self.traj_number})")
                self._reset_all_interceptors()
            return self.prev_phi_cmd

        self.missed_frames = 0
        xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            return self.prev_phi_cmd

        self.ransac_fitter.update(timestamp, xyz)
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        intercept = self.ransac_fitter.predict_interception(timestamp)
        if intercept is None:
            print(f"[ransac] waiting for min points ({self.ransac_fitter._t.size}/{self.ransac_fitter.min_points})")
            self.interception_point_ROBOT = None
            self.interception_time = None
            return self.prev_phi_cmd

        print(f"[ransac] intercept={np.round(intercept, 3)}, inliers={self.ransac_fitter._n_inliers}")
        self.interception_point_ROBOT = intercept

        phi_cmd = self._resolve_ik(intercept)
        if phi_cmd is None:
            print("[ransac] IK failed for intercept point")
            return self.prev_phi_cmd

        return self._apply_phi_cmd(phi_cmd, intercept)

    # ── NEW: Approach 5 — EWMA velocity estimator ─────────────────────────

    def ballXYZ_to_phi_cmd_ewma(
        self,
        XYZ: np.ndarray,
        ball_found: bool,
        timestamp: float,
    ) -> np.ndarray:
        """
        Intercept using an exponentially-weighted moving average velocity
        estimate plus ballistic plane intersection.

        The simplest viable approach: O(1) memory and CPU. Works well for
        smooth, slow-to-medium speed trajectories at high frame rates.
        Tune alpha in EWMAInterceptor.__init__ (higher = faster response,
        noisier; lower = smoother, slower to react).
        """
        if not ball_found:
            self.missed_frames += 1
            if self.missed_frames >= self.missed_frames_max:
                print(f"[ewma] tracking lost, resetting (traj #{self.traj_number})")
                self._reset_all_interceptors()
            return self.prev_phi_cmd

        self.missed_frames = 0
        xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            return self.prev_phi_cmd

        self.ewma_interceptor.update(timestamp, xyz)
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        intercept = self.ewma_interceptor.predict_interception(timestamp)
        if intercept is None:
            print("[ewma] waiting for enough observations")
            self.interception_point_ROBOT = None
            self.interception_time = None
            return self.prev_phi_cmd

        print(f"[ewma] intercept={np.round(intercept, 3)}")
        self.interception_point_ROBOT = intercept

        phi_cmd = self._resolve_ik(intercept)
        if phi_cmd is None:
            print("[ewma] IK failed for intercept point")
            return self.prev_phi_cmd

        return self._apply_phi_cmd(phi_cmd, intercept)

    # ── hardware methods (all unchanged from original) ────────────────────

    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None) -> None:
        input_phi_cmd = np.asarray(phi_Cmd, dtype=np.float64)
        if input_phi_cmd.shape != (4,):
            raise ValueError("phi_Cmd must be an iterable of 4 joint angles [rad].")

        r = 0.05

        if np.equal(self.phi_cmd, input_phi_cmd).all():
            print(".", end="")
            return
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
            print("Invalid phi_cmd: ", self.phi_cmd)
            return

        self.myArm.read_write_std(phiCMD=self.phi_cmd, grpCMD=self._gripper, baseLED=self._led)
        print("commanded to phi: ", self.phi_cmd, "pos: ", self.pos_cmd)

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
        """QUANSER_ARM_FPK v 1.0 - 30th August 2020"""
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
        """QUANSER_ARM_IPK v 1.0 - 31st August 2020"""
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
            sin_term = (H - N * cos_term) / (M)
            j2 = np.arctan2(sin_term, cos_term)
            return j2

        A, C, H, D1, D2, F = inv_kin_setup(p)

        theta[2, 0] = 2 * np.arctan2(C + np.sqrt(C**2 - F**2), F)
        theta[2, 1] = 2 * np.arctan2(C - np.sqrt(C**2 - F**2), F)
        theta[2, 2] = 2 * np.arctan2(C + np.sqrt(C**2 - F**2), F)
        theta[2, 3] = 2 * np.arctan2(C - np.sqrt(C**2 - F**2), F)

        theta[1, 0] = solve_case_C_j2(theta[2, 0], A, C, D1, H)
        theta[1, 1] = solve_case_C_j2(theta[2, 1], A, C, D1, H)
        theta[1, 2] = solve_case_C_j2(theta[2, 2], A, C, D2, H)
        theta[1, 3] = solve_case_C_j2(theta[2, 3], A, C, D2, H)

        # fmt: off
        theta[0, 0] = np.arctan2( p[1]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 0] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 0] + theta[2, 0] ) ) ,
                                  p[0]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 0] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 0] + theta[2, 0] ) ) )
        theta[0, 1] = np.arctan2( p[1]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 1] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 1] + theta[2, 1] ) ) ,
                                  p[0]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 1] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 1] + theta[2, 1] ) ) )
        theta[0, 2] = np.arctan2( p[1]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 2] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 2] + theta[2, 2] ) ) ,
                                  p[0]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 2] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 2] + theta[2, 2] ) ) )
        theta[0, 3] = np.arctan2( p[1]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 3] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 3] + theta[2, 3] ) ) ,
                                  p[0]/( self.myArmUtilities.LAMBDA_2 * np.cos( theta[1, 3] ) - (self.myArmUtilities.LAMBDA_3 + self.L_6) * np.sin( theta[1, 3] + theta[2, 3] ) ) )
        # fmt: on

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
