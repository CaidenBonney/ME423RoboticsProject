# Standard library imports
import time
import numpy as np
import importlib.util
import os
from typing import Optional

# Load the modified QArm object (changed from original)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../QarmHardwareFiles/AdjustedQarmHardwareFiles/qarm.py"))
spec = importlib.util.spec_from_file_location("qarm", module_path)
qarm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qarm)

from hal.products.qarm import QArmUtilities
from Trajectory import Trajectory
from Ballistic import BallisticInterceptor



class Arm:
    def __init__(self) -> None:
        # Internal variables from basic position mode py files
        self.startTime = time.time()
        self.sampleRate = 1000  # [Hz]
        self.sampleTime = 1 / self.sampleRate
        self.myArm = qarm.QArm(hardware=1)
        self.myArmUtilities = QArmUtilities()

        # Internal variables for current state of the arm
        self.prev_meas_phi = np.array([0, 0, 0, 0], dtype=np.float64) # [rad] joint angles for 4 joints in order: base, shoulder, elbow, wrist
        self.prev_meas_pos = np.array([0, 0, 0], dtype=np.float64) # [m] With respect to base frame
        self.phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64) # [rad] joint angles for 4 joints in order: base, shoulder, elbow, wrist
        self.pos_cmd = np.array([0, 0, 0], dtype=np.float64) # [m] With respect to base frame
        self._phi_dot = np.array([0, 0, 0, 0], dtype=np.float64) # [rad/s]
        self._gripper = np.array(0.0, dtype=np.float64) # 0 = open, 1 = closed
        self._led = np.array([1, 0, 1], dtype=np.float64) # [R, G, B] values as floats from 0 to 1

        self._R = np.identity(3, dtype=np.float64) # rotation matrix from end-effector frame to base frame
        
        # rolling buffer for trajectory
        self._pos_q_max = 100 # maximum number of points in trajectory
        self._pos_q = np.empty((self._pos_q_max, 3), dtype=np.float64) # [m] With respect to base frame
        self._time_q = np.empty(self._pos_q_max, dtype=np.float64) # [ms] time stamps for each point in trajectory

        # Interceptor implementations

        ## Trajectory class is out of date but kept for legacy code
        self.traj = Trajectory()

        ## Ballistic interceptor is the most accurate
        self.fixedX = 0.6 # [m] the x-coordinate in base frame for fixed catching plane
        self._catch_z = 0.30   # [m] the z-height in base frame for fixed catching plane
        self.ballistic_interceptor   = BallisticInterceptor(catch_z=self._catch_z)

        ## Trajectory refresh variables
        self.traj_number = 0 # Number of trajectories created, used for debugging
        self.missed_frames = 0 # Count the number of frames since the ball was last seen
        self.missed_frames_max = 20 # Maximum number of missed frames before resetting the trajectory

        ## Interception variables
        self.prev_phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64) 
        self.interception_point_ROBOT = None # [m] Interception point calculated w.r.t. base frame
        self.interception_time = None # [ms] time of interception w.r.t to first timestamp in interceptor class

        # Transformation variables
        self.T04 = np.identity(4, dtype=np.float64) # Transformation matrix from base frame to adjusted end effector frame
        self.L_6 = 0.25 # [m] Distance added for net end effector center

        # Send arm to home configuration
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


        # Calculate offset from home position. Measuring joints is zero on startup. Offest used for proper position readings
        # Note if you write [0,0,0,0], the arm will move to the home position, despite at startup the arm is not at home and reads [0,0,0,0] in the shifted position
        # THis means reading and writing are not synchronized
        self._phi_offset = np.array(self.myArm.measJointPosition[0:4], dtype=np.float64, copy=True)

    def _reset_all_interceptors(self) -> None:
        """Reset every interceptor and the Trajectory on tracking loss. 
        More useful when multiple trajectories were being tested"""
        self.traj = Trajectory()
        self.traj_number += 1 # Increment trajectory number
        self.ballistic_interceptor.reset()

    def _resolve_ik(self, target_xyz: np.ndarray) -> Optional[np.ndarray]:
        """
        Run Inverse Kinematics (IK) for target_xyz and return the joint-angle solution closest to
        the current arm configuration, or None if IK fails.
        """
        phi_seed = np.asarray(self.phi, dtype=np.float64) # Work with a copy of current joint angles
        try:
            # Run IK
            all_solns, ik_soln = self.qarm_inverse_kinematics(target_xyz, 0, phi_seed)
        except Exception as e:
            print(f"IK exception: {e}")
            return None

        # Reformat solutions
        all_solns = np.asarray(all_solns, dtype=np.float64)
        chosen_phi = None

        # If solutions are in proper shape
        if all_solns.ndim == 2 and all_solns.shape[0] == 4:
            # Check for valid solutions
            valid_cols = np.all(np.isfinite(all_solns), axis=0)
            if np.any(valid_cols):
                # Calc the closest solution
                valid_solns = all_solns[:, valid_cols].T
                idx = np.argmin(np.linalg.norm(valid_solns - phi_seed, axis=1))
                chosen_phi = valid_solns[idx]

        # If checking from all solutions failed, use the fallback solution
        if chosen_phi is None:
            fallback = np.asarray(ik_soln, dtype=np.float64).reshape(-1)
            if fallback.shape == (4,) and np.all(np.isfinite(fallback)):
                chosen_phi = fallback

        return chosen_phi

    def _apply_phi_cmd(self, phi_cmd: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        """Store pos_cmd/prev_phi_cmd and return phi_cmd for the worker to pass to move().

        Critically does NOT set self.phi_cmd — that is move()'s job.  If we set
        self.phi_cmd here, move() sees an identical value on the very next call
        and returns early without sending the hardware command (the dot guard).
        """
        phi_cmd = np.asarray(phi_cmd, dtype=np.float64)
        self.pos_cmd = intercept.copy()
        self.prev_phi_cmd = phi_cmd.copy()
        return phi_cmd

    # ── original methods (unchanged) ──────────────────────────────────────

    def elapsed_time(self) -> float:
        """"time elapsed since arm was initialized"""
        return time.time() - self.startTime

    def print_measurement_check(self, label: str = "measurement check") -> None:
        """Prints raw and offset-corrected joint measurements for debugging."""
        self.myArm.read_std()
        raw_phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)
        joint_speed = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        rel_phi = raw_phi - self._phi_offset
        print(label)
        print("raw measured phi:", raw_phi)
        print("stored home offset:", self._phi_offset)
        print("home-relative phi:", rel_phi)
        print("measured joint speed:", joint_speed)

    # THIS IS NOT USED, but referenced in legacy code and tests
    def ballXYZ_to_phi_cmd(self, XYZ: np.ndarray, ball_found: bool, timestamp: float) -> Optional[np.ndarray]:
        """Original polynomial-fit method (UNUSED)."""

        # If the ball was not found, increment missed frames
        if not ball_found or ball_found is None:
            self.missed_frames += 1
            # If the number of missed frames exceeds the maximum, reset the trajectory
            if self.missed_frames == self.missed_frames_max:
                print("Creating new trajectory object, lost tracking", self.traj_number, "time: ", timestamp)
                self.traj = Trajectory()
                self.traj_number += 1
            # if ball not found, return the previous command
            return self.prev_phi_cmd

        # If the ball was found, reset missed frames
        self.missed_frames = 0

        # Convert input to a clean (3,) float vector; reject NaN/Inf early.
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        # update trajectory object with current measurement
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        # If the trajectory is not long enough, return the previous command
        if self.traj.t.size < 3:
            print("Not enough points to fit parabola, using most recent measurement for IK.")
            return self.prev_phi_cmd
        else:
            print("Fitting parabola,-----------------------------------------")

        # Calculate the roots of the z-equation at the catch plane
        pz = self.traj.pz.copy()
        pz[-1] -= self._catch_z
        z_roots = np.roots(pz)
        real_dt = z_roots[np.abs(z_roots.imag) < 1e-8].real

        # Calculate the time shift from the current time to the future hit
        t_now_shift = timestamp - self.traj.t0
        fut_dt = real_dt[real_dt >= t_now_shift]

        # If no future hits were found, return the previous command
        if fut_dt.size == 0:
            self.interception_point_ROBOT = None
            self.interception_time = None
            # print ("No future hits found, returning previous command")
            return self.prev_phi_cmd

        # Calculate the time of the future hit
        t_hit_ms = self.traj.t0[0] + np.min(fut_dt)
        print(f"Future hit: {t_hit_ms + timestamp}, fut_dt: {fut_dt}")

        # Calculate the position of the future hit
        ik_xyz = self.traj.predict_pos(t_hit_ms)[:, 0]
        self.interception_point_ROBOT = ik_xyz
        self.interception_time = t_hit_ms

        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(ik_xyz, 0, self.phi)

        # Find the closest solution
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
        """Fixed-XZ plane interception (used for test purposes)."""

        # Convert input to a clean (3,) float vector; reject NaN/Inf early.
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        # Update the fixed x and z coordinates
        xyz_meas[0] = fixed_x
        xyz_meas[2] = fixed_z

        if not ball_found or ball_found is None:
            return self.prev_phi_cmd

        self.interception_point_ROBOT = xyz_meas
        self.interception_time = timestamp

        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(xyz_meas, 0, self.phi)

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


    def ballXYZ_to_phi_cmd_ballistic(
        self,
        XYZ: np.ndarray,
        ball_found: bool,
        timestamp: float,
    ) -> np.ndarray:
        """
        Intercept using physics-based ballistic prediction.
        Fits the trajectory with gravity as a fixed constant, meaning closed-form catch-plane intersection
        Set catch_z in __init__ above.
        """

        # If the ball was not found, increment missed frames
        if not ball_found:
            self.missed_frames += 1
            # If the number of missed frames exceeds the maximum, reset the trajectory
            if self.missed_frames >= self.missed_frames_max:
                print(f"[ballistic] tracking lost, resetting (traj #{self.traj_number})")
                self._reset_all_interceptors()
            return self.prev_phi_cmd

        # If the ball was found, reset missed frames
        self.missed_frames = 0

        # Convert input to a clean (3,) float vector; reject NaN/Inf early.
        xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            return self.prev_phi_cmd

        # Update the ballistic interceptor with the current measurement
        self.ballistic_interceptor.update(timestamp, xyz)
        # Update the trajectory with the current measurement (OUT OF DATE)
        self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

        # Predict the interception point
        intercept = self.ballistic_interceptor.predict_interception(timestamp)

        # If no interception point was found, MIRROR the fixed x and z coordinates
        if intercept is None:
            print("[ballistic] waiting for enough points or no future root")
            self.interception_point_ROBOT = None
            self.interception_time = None
            # return self.prev_phi_cmd
            intercept = np.array([self.fixedX, XYZ[1], self._catch_z], dtype=np.float64)

        # Otherwise, log the interception point
        else:
            print(f"[ballistic] intercept={np.round(intercept, 3)}")
            self.interception_point_ROBOT = intercept

        # Resolve the inverse kinematics solution for the interception point
        phi_cmd = self._resolve_ik(intercept)
        if phi_cmd is None:
            print("[ballistic] IK failed for intercept point")
            return self.prev_phi_cmd

        # Apply the phi command to the arm
        return self._apply_phi_cmd(phi_cmd, intercept)


    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None) -> None:
        """Check intputs and send to arm.
        
        Args:
            phi_Cmd: Desired joint angles [rad].
            gripper_Cmd: Desired gripper state (0=open, 1=closed).
            led_Cmd: Desired LED state [R, G, B], each in the range [0, 1].
        """

        # Check input phi command
        input_phi_cmd = np.asarray(phi_Cmd, dtype=np.float64)
        if input_phi_cmd.shape != (4,):
            raise ValueError("phi_Cmd must be an iterable of 4 joint angles [rad].")

        # if the input command is the same as the previous command, do nothing
        if np.equal(self.phi_cmd, input_phi_cmd).all():
            # print(".", end="")
            return
        else:
            self.phi_cmd = input_phi_cmd

        # Check input gripper command
        if gripper_Cmd is not None:
            gripper_cmd = float(gripper_Cmd)
            if not (0.0 <= gripper_cmd <= 1.0):
                raise ValueError("gripper_Cmd must be between 0 and 1.")
            self._gripper = np.asarray(gripper_cmd, dtype=np.float64)

        # Check input led command
        if led_Cmd is not None:
            led_cmd = np.asarray(led_Cmd, dtype=np.float64)
            if led_cmd.shape != (3,):
                raise ValueError("led_Cmd must be an iterable of 3 values [R, G, B].")
            if np.any((led_cmd < 0) | (led_cmd > 1)):
                raise ValueError("Each led_Cmd value must be between 0 and 1.")
            self._led = led_cmd

        try:
            # Check joint limits and workspace
            self.limit_check(self.phi_cmd)
            self.workspace_check(self.phi_cmd)
        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Invalid phi_cmd: ", self.phi_cmd)
            return

        # Send command to arm
        self.myArm.read_write_std(phiCMD=self.phi_cmd, grpCMD=self._gripper, baseLED=self._led)
        print("commanded to phi: ", self.phi_cmd, "pos: ", self.pos_cmd)

    def limit_check(self, phi_cmd) -> None:
        """Check joint limits and raise ValueError if exceeded."""
        if phi_cmd[0] < -np.radians(170) or phi_cmd[0] > np.radians(170):
            raise ValueError("Base Phi limit reached.")
        elif phi_cmd[1] < -np.radians(85) or phi_cmd[1] > np.radians(85):
            raise ValueError("Shoulder Phi limit reached.")
        elif phi_cmd[2] < -np.radians(95) or phi_cmd[2] > np.radians(75):
            raise ValueError("Elbow Phi limit reached.")
        elif phi_cmd[3] < -np.radians(160) or phi_cmd[3] > np.radians(160):
            raise ValueError("Wrist Phi limit reached.")

    def workspace_check(self, phi_cmd) -> None:
        """Check workspace limits and raise ValueError if exceeded."""
        self.pos_cmd, _ = self.qarm_forward_kinematics(phi_cmd)
        if self.pos_cmd[2] < 0.1:
            raise ValueError("End-effector z position limit reached.")

    def home(self) -> None:
        """Home the arm."""
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
