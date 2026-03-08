# Special QArm library imports
from typing import Optional

from pal.products.qarm import QArm  # pyright: ignore[reportMissingImports]
from hal.products.qarm import QArmUtilities  # pyright: ignore[reportMissingImports]

# Standard library imports
import time
import numpy as np
from Trajectory import update_trajectory


class Arm:
    def __init__(self) -> None:
        # Internal variables from basic position mode py files
        self.startTime = time.time()
        self.sampleRate = 200
        self.sampleTime = 1 / self.sampleRate
        self.myArm = QArm(hardware=1)
        self.myArmUtilities = QArmUtilities()
        # print("Sample Rate is ", self.sampleRate, " Hz. Simulation will run until you type Ctrl+C to exit.")

        # Internal variables for current state of the arm
        # [rad] joint angles for 4 joints in order: base, shoulder, elbow, wrist
        self._phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self._phi_dot = np.array([0, 0, 0, 0], dtype=np.float64)  # [rad/s]
        self._position = np.array([0, 0, 0], dtype=np.float64)  # [m]
        self._gripper = np.float64(0.0)  # [0 = open, 1 = closed]
        self._led = np.array([0, 0, 0], dtype=np.float64)  # [R, G, B] values as floats from 0 to 1

        self._xyz_origin_offset = np.array([0.45, 0.0, 0.49], dtype=np.float64)
        self._pos_q_max = 64
        self._pos_q = np.empty((self._pos_q_max, 3), dtype=np.float64)
        self._time_q = np.empty(self._pos_q_max, dtype=np.float64)
        self._q_write_idx = 0
        self._q_count = 0

        self.home()
        # [rad] joint angles for 4 joints in order: base, shoulder, elbow, wrist
        self._phi_offset = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)

    def elapsed_time(self) -> float:
        return time.time() - self.startTime

    def XYZ_to_phi_cmd(
        self, XYZ: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.float64], Optional[np.ndarray]]:
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        t_now_s = self.elapsed_time()

        self._pos_q[self._q_write_idx, :] = xyz_meas
        self._time_q[self._q_write_idx] = t_now_s
        self._q_write_idx = (self._q_write_idx + 1) % self._pos_q_max
        if self._q_count < self._pos_q_max:
            self._q_count += 1

        ik_xyz = xyz_meas
        if self._q_count >= 3:
            oldest_idx = (self._q_write_idx - self._q_count) % self._pos_q_max
            order_idx = (oldest_idx + np.arange(self._q_count, dtype=np.int64)) % self._pos_q_max
            t_hist = self._time_q[order_idx]
            pos_hist = self._pos_q[order_idx, :]

            traj = update_trajectory(t_hist, pos_hist, self._pos_q_max)
            pz = traj.pz.copy()
            pz[-1] -= 0.49  # plane of intersection is at z=-0.49
            z_roots = np.roots(pz)
            real_dt = z_roots[np.abs(z_roots.imag) < 1e-8].real
            fut_dt = real_dt[real_dt >= (t_hist[-1] - traj.t0)]

            if fut_dt.size > 0:
                t_hit_s = traj.t0 + np.min(fut_dt)
            elif real_dt.size > 0:
                t_hit_s = traj.t0 + np.max(real_dt)
            else:
                t_hit_s = t_hist[-1]

            ik_xyz = traj.pos(t_hit_s)[:, 0]

        ik_pos_cmd = ik_xyz - self._xyz_origin_offset

        ik_all_solns, ik_soln = self.myArmUtilities.qarm_inverse_kinematics(
            ik_pos_cmd, 0, self.myArm.measJointPosition[0:4]
        )
        phi_seed = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)
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

        phi_cmd = np.asarray(chosen_phi, dtype=np.float64)
        gripper_cmd = np.float64(0.0)
        led_cmd = None

        return phi_cmd, gripper_cmd, led_cmd

    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None) -> None:
        # region: Process Inputs:
        phi_cmd = np.asarray(phi_Cmd, dtype=np.float64)
        if phi_cmd.shape != (4,):
            raise ValueError("phi_Cmd must be an iterable of 4 joint angles [rad].")
        self._phi = phi_cmd

        # Optional gripper command
        if gripper_Cmd is not None:
            gripper_cmd = float(gripper_Cmd)
            if gripper_cmd < 0 or gripper_cmd > 1:
                raise ValueError("gripper_Cmd must be a value between 0 and 1.")
            self._gripper = np.float64(gripper_cmd)

        # Optional led command
        if led_Cmd is not None:
            led_cmd = np.asarray(led_Cmd, dtype=np.float64)
            if led_cmd.shape != (3,):
                raise ValueError("led_Cmd must be an iterable of 3 values [R, G, B].")
            if np.any((led_cmd < 0) | (led_cmd > 1)):
                raise ValueError("Each led_Cmd value must be between 0 and 1.")
            self._led = led_cmd
        # endregion

        # Check limits and workspace before sending command to arm will raise ValueError if checks fail
        self.limit_check(phi_cmd)
        self.workspace_check(phi_cmd)

        # Commands arm to move to desired phi_cmd with gripper and LED states.
        # Note that speed is not specified in this command.
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

    # Only checks physical limits of the arm
    def limit_check(self, phi_cmd) -> None:
        #   Phi limits (from QArm documentation):
        #       Base: ± 170 deg
        #       Shoulder: ± 85 deg
        #       Elbow: -95 deg / +75 deg
        #       Wrist: ± 160 deg

        if self._phi[0] < -np.radians(170) or self._phi[0] > np.radians(170):
            raise ValueError("Base Phi limit reached. Arm moved to home position.")
        elif self._phi[1] < -np.radians(85) or self._phi[1] > np.radians(85):
            raise ValueError("Shoulder Phi limit reached. Arm moved to home position.")
        elif self._phi[2] < -np.radians(95) or self._phi[2] > np.radians(75):
            raise ValueError("Elbow Phi limit reached. Arm moved to home position.")
        elif self._phi[3] < -np.radians(160) or self._phi[3] > np.radians(160):
            raise ValueError("Wrist Phi limit reached. Arm moved to home position.")

    # Checks if the arm will run into the table
    def workspace_check(self, phi_cmd) -> None:
        # TODO: Update workspace limits to real values and test phi_cmd to make sure it won't run into the table

        # Base parrallel to table so impossible to run into table with any base angle
        if self._phi[1] < -np.radians(85) or self._phi[1] > np.radians(85):
            raise ValueError("Shoulder Workspace limit reached. Arm moved to home position.")
        elif self._phi[2] < -np.radians(95) or self._phi[2] > np.radians(75):
            raise ValueError("Elbow Workspace limit reached. Arm moved to home position.")
        elif self._phi[3] < -np.radians(160) or self._phi[3] > np.radians(160):
            raise ValueError("Wrist Workspace limit reached. Arm moved to home position.")

    def home(self) -> None:
        self._phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self._led = np.array([1, 0, 0], dtype=np.float64)
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

    @property
    def phi(self):
        # Update phi to current state of the arm
        self._phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64) - self._phi_offset
        return self._phi

    @property
    def position(self) -> np.ndarray:
        # Update phi to current state of the arm and then calculate position with forward kinematics
        self._position = self.myArmUtilities.qarm_forward_kinematics(self.phi)  # this will update self._phi as well
        return self._position

    @property
    def gripper(self) -> np.float64:
        return self._gripper

    @property
    def led(self) -> np.ndarray:
        return self._led

    @property
    def phi_dot(self) -> np.ndarray:
        # Update phi_dot to current state of the arm
        self._phi_dot = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        return self._phi_dot
