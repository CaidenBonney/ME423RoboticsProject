# Special QArm library imports
from typing import Optional

from pal.products.qarm import QArm  # pyright: ignore[reportMissingImports]
from hal.products.qarm import QArmUtilities  # pyright: ignore[reportMissingImports]

# Standard library imports
import time
import numpy as np
from Trajectory import Trajectory


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
        self._R = np.identity(3, dtype=np.float64)  # rotation matrix from end-effector frame to base frame
        self._gripper = np.array(0.0, dtype=np.float64)  # 0 = open, 1 = closed
        self._led = np.array([1, 0, 1], dtype=np.float64)  # [R, G, B] values as floats from 0 to 1

        # self._xyz_origin_offset = np.array([0.45, 0.0, 0.49], dtype=np.float64)
        self._pos_q_max = 64
        self._pos_q = np.empty((self._pos_q_max, 3), dtype=np.float64)
        self._time_q = np.empty(self._pos_q_max, dtype=np.float64)
        self._q_write_idx = 0
        self._q_count = 0

        self.traj = Trajectory() # initialize empty trajectory
        self.missed_frames = 0 # count the number of frames since the ball was last seen
        self.missed_frames_max = 10
        self.prev_phi_cmd = np.array([0, 0, 0, 0], dtype=np.float64)

        # transformation matrix from end-effector frame to base frame adjusted in qarm_forward_kinematics
        self.T04 = np.identity(4, dtype=np.float64)
        self.L_6 = 0.075  # distance from qarm end-effector center to net end effector center in meters

        self.home()
        # [rad] joint angles for 4 joints in order: base, shoulder, elbow, wrist
        self._phi_offset = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)
        print(self._phi_offset)

    def elapsed_time(self) -> float:
        return time.time() - self.startTime

    def ballXYZ_to_phi_cmd(self, XYZ: np.ndarray, ball_found: bool, timestamp: float) -> Optional[np.ndarray]:
        if not ball_found:
            self.missed_frames += 1
            if self.missed_frames == self.missed_frames_max:
                print("Creating new trajectory object, lost tracking")
                self.traj = Trajectory()
            return self.prev_phi_cmd
        # Convert input to a clean (3,) float vector; reject NaN/Inf early.
        self.missed_frames = 0
        xyz_meas = np.asarray(XYZ, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz_meas)):
            raise ValueError("XYZ command contains NaN/Inf values.")

        t_now_s = self.elapsed_time()  # current time (seconds)

        # Ring-buffer append: store the newest XYZ + timestamp into circular history arrays.
        self._pos_q[self._q_write_idx, :] = xyz_meas
        self._time_q[self._q_write_idx] = t_now_s
        self._q_write_idx = (self._q_write_idx + 1) % self._pos_q_max
        if self._q_count < self._pos_q_max:
            self._q_count += 1

        # Default IK target is the most recent measurement.
        ik_xyz = xyz_meas

        # If we have enough samples, fit a trajectory to recent XYZ history and predict
        # where it will be when it crosses a specific z-plane (z = 0.49 in your code).
        if self._q_count >= 3:
            oldest_idx = (self._q_write_idx - self._q_count) % self._pos_q_max
            order_idx = (oldest_idx + np.arange(self._q_count, dtype=np.int64)) % self._pos_q_max
            t_hist = self._time_q[order_idx]  # ordered times (oldest -> newest)
            pos_hist = self._pos_q[order_idx, :]  # ordered positions (oldest -> newest)

            self.traj.update_trajectory(timestamp, XYZ, self._pos_q_max)

            # Solve for times when the fitted z(t) hits the plane z = 0.49.
            pz = self.traj.pz.copy()
            pz[-1] -= 0.49  # plane of intersection is at z= 0.49
            z_roots = np.roots(pz)

            # Keep real roots only, then prefer future intersections if any exist.
            real_dt = z_roots[np.abs(z_roots.imag) < 1e-8].real
            fut_dt = real_dt[real_dt >= (t_hist[-1] - self.traj.t0)]

            case = 1 # trace logic for prediction
            if fut_dt.size > 0:
                t_hit_s = self.traj.t0 + np.min(fut_dt)  # soonest future hit
            elif real_dt.size > 0:
                t_hit_s = self.traj.t0 + np.max(real_dt)  # most recent past hit
                case = 2
            else:
                t_hit_s = t_hist[-1]  # no roots -> just use "now"
                case = 3

            ik_xyz = self.traj.predict_pos(t_hit_s + self.traj.t0)[:, 0]  # predicted XYZ at the selected time
            print("ik_xyz: ", ik_xyz, "t_hit_s: ", t_hit_s, "t0: ", self.traj.t0, "case: ", case)

        # Final frame for IK input is the predicted XYZ.
        ik_pos_cmd = ik_xyz
        # print(ik_pos_cmd)

        # Compute IK: all candidate solutions + a fallback solution.
        ik_all_solns, ik_soln = self.qarm_inverse_kinematics(ik_pos_cmd, 0, self.phi)

        phi_seed = np.asarray(self.phi, dtype=np.float64)  # current joint angles
        all_solns = np.asarray(ik_all_solns, dtype=np.float64)
        chosen_phi = None

        # Prefer a valid solution that is closest to the current joint configuration
        # (avoids elbow-up/down "flips" when multiple IK solutions exist).
        if all_solns.ndim == 2 and all_solns.shape[0] == 4:
            valid_cols = np.all(np.isfinite(all_solns), axis=0)
            # If any valid solutions exist, choose the one closest to current joint angles.
            if np.any(valid_cols):
                valid_solns = all_solns[:, valid_cols].T
                idx = np.argmin(np.linalg.norm(valid_solns - phi_seed, axis=1))
                chosen_phi = valid_solns[idx]

        # Fallback: use the solver-provided single solution if the multi-solution path failed.
        if chosen_phi is None:
            fallback_phi = np.asarray(ik_soln, dtype=np.float64).reshape(-1)
            if fallback_phi.shape == (4,) and np.all(np.isfinite(fallback_phi)):
                chosen_phi = fallback_phi

        # If no finite 4-joint solution exists, the target is unreachable (or frames are wrong).
        if chosen_phi is None:
            raise ValueError("IK failed: target appears unreachable.")

        phi_cmd = np.asarray(chosen_phi, dtype=np.float64)
        self.prev_phi_cmd = phi_cmd
        return phi_cmd

    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None) -> None:
        # region: Process Inputs:
        phi_cmd = np.asarray(phi_Cmd, dtype=np.float64)
        if phi_cmd.shape != (4,):
            raise ValueError("phi_Cmd must be an iterable of 4 joint angles [rad].")

        # If the current command is the same as the previous command, do nothing
        if np.equal(phi_cmd, self._phi).all():
            return  # no movement needed
        else:
            self._phi = phi_cmd

        # Optional gripper command
        if gripper_Cmd is not None:
            gripper_cmd = float(gripper_Cmd)
            if not (0.0 <= gripper_cmd <= 1.0):
                raise ValueError("gripper_Cmd must be between 0 and 1.")
            self._gripper = np.asarray(gripper_cmd, dtype=np.float64)

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
        # Currently uses initial internal values for gripper and LED if not specified as an input.
        # Note that speed is not specified in this command.
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

    # Only checks physical limits of the arm
    def limit_check(self, phi_cmd) -> None:
        #   Phi limits (from QArm documentation):
        #       Base:         ± 170 deg
        #       Shoulder:     ± 85  deg
        #       Elbow:    -95 deg / +75 deg
        #       Wrist:        ± 160 deg

        if phi_cmd[0] < -np.radians(170) or phi_cmd[0] > np.radians(170):
            raise ValueError("Base Phi limit reached. Arm moved to home position.")
        elif phi_cmd[1] < -np.radians(85) or phi_cmd[1] > np.radians(85):
            raise ValueError("Shoulder Phi limit reached. Arm moved to home position.")
        elif phi_cmd[2] < -np.radians(95) or phi_cmd[2] > np.radians(75):
            raise ValueError("Elbow Phi limit reached. Arm moved to home position.")
        elif phi_cmd[3] < -np.radians(160) or phi_cmd[3] > np.radians(160):
            raise ValueError("Wrist Phi limit reached. Arm moved to home position.")

    # Checks if the arm will run into the table
    def workspace_check(self, phi_cmd) -> None:

        pos_cmd, _ = self.qarm_forward_kinematics(phi_cmd)  # updates self.T04 based on phi_cmd
        # Check z position of end-effector to make sure it won't run into the table.
        if pos_cmd[2] < 0.1:  # keeps z position from being within 0.1 meters of the table
            raise ValueError("End-effector z position limit reached. Arm moved to home position.")

    def home(self) -> None:
        self._phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self._led = np.array([1, 0, 0], dtype=np.float64)
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

    def qarm_forward_kinematics(self, phi):
        """QUANSER_ARM_FPK v 1.0 - 30th August 2020

        REFERENCE:
        Chapter 3. Forward Kinematics
        Robot Dynamics and Control, Spong, Vidyasagar, 1989

        INPUTS:
        phi     : Alternate joint angles vector 4 x 1

        OUTPUTS:
        p4      : End-effector frame {4} position vector expressed in base frame {0}
        R04     : rotation matrix from end-effector frame {4} to base frame {0}"""

        # From phi space to theta space
        theta = phi.copy()
        theta[0] = phi[0]
        theta[1] = phi[1] + self.myArmUtilities.BETA - np.pi / 2
        theta[2] = phi[2] - self.myArmUtilities.BETA
        theta[3] = phi[3]

        # Transformation matrices for all frames:

        # T{i-1}{i} = quanser_arm_DH(  a, alpha,  d,     theta )
        T01 = self.myArmUtilities.quanser_arm_DH(0, -np.pi / 2, self.myArmUtilities.LAMBDA_1, theta[0])
        T12 = self.myArmUtilities.quanser_arm_DH(self.myArmUtilities.LAMBDA_2, 0, 0, theta[1])
        T23 = self.myArmUtilities.quanser_arm_DH(0, -np.pi / 2, 0, theta[2])
        T34 = self.myArmUtilities.quanser_arm_DH(0, 0, self.myArmUtilities.LAMBDA_3 + self.L_6, theta[3])

        T02 = T01 @ T12
        T03 = T02 @ T23
        self.T04 = T03 @ T34

        # Position of end-effector Transformation

        # Extract the Position vector
        # p1   = T01(1:3,4);
        # p2   = T02(1:3,4);
        # p3   = T03(1:3,4);
        p4 = self.T04[0:3, 3]

        # Extract the Rotation matrix
        # R01 = T01(1:3,1:3);
        # R02 = T02(1:3,1:3);
        # R03 = T03(1:3,1:3);
        R04 = self.T04[0:3, 0:3]

        return p4, R04

    def qarm_inverse_kinematics(self, p, gamma, phi_prev):
        """
        QUANSER_ARM_IPK v 1.0 - 31st August 2020

        REFERENCE:
            Chapter 4. Inverse Kinematics
            Robot Dynamics and Control, Spong, Vidyasagar, 1989

        INPUTS:
            p: end-effector position vector expressed in base frame {0}
            gamma: wrist rotation angle gamma

        OUTPUTS:
            phiOptimal : Best solution depending on phi_prev
            phi: All four Inverse Kinematics solutions as a 4x4 matrix. Each col is a solution.
        """

        # Initialization
        theta = np.zeros((4, 4), dtype=np.float64)
        phi = np.zeros((4, 4), dtype=np.float64)

        # Equations:
        # LAMBDA_2 cos(theta2) + (-LAMBDA_3) sin(theta2 + theta3) = sqrt(x^2 + y^2)
        #   A     cos( 2    ) +     C      sin(   2   +    3  ) =    D

        # LAMBDA_2 sin(theta2) - (-LAMBDA_3) cos(theta2 + theta3) = LAMBDA_1 - z
        #   A     sin( 2    ) -     C      cos(   2   +    3  ) =    H

        # Solution:
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

        # Joint 3 solution:
        theta[2, 0] = 2 * np.arctan2(C + np.sqrt(C**2 - F**2), F)
        theta[2, 1] = 2 * np.arctan2(C - np.sqrt(C**2 - F**2), F)
        theta[2, 2] = 2 * np.arctan2(C + np.sqrt(C**2 - F**2), F)
        theta[2, 3] = 2 * np.arctan2(C - np.sqrt(C**2 - F**2), F)

        # Joint 2 solution:
        theta[1, 0] = solve_case_C_j2(theta[2, 0], A, C, D1, H)
        theta[1, 1] = solve_case_C_j2(theta[2, 1], A, C, D1, H)
        theta[1, 2] = solve_case_C_j2(theta[2, 2], A, C, D2, H)
        theta[1, 3] = solve_case_C_j2(theta[2, 3], A, C, D2, H)

        # Joint 1 solution:
        # turns off formatter for section of code with complex equations for better readability
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
        # turns on formatter for after section of code with complex equations for better readability

        # Remap theta back to phi
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
        self.myArm.read_std()  # updates self.myArm.measJointPosition
        # Update phi to current state of the arm
        # print("trying to get phi")
        self._phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64) - self._phi_offset
        return self._phi

    @property
    def position(self) -> np.ndarray:
        # Update phi to current state of the arm and then calculate position with forward kinematics
        self._position, self._R = self.qarm_forward_kinematics(self.phi)  # note this will update self._phi as well
        return self._position

    @property
    def R(self) -> np.ndarray:
        return self._R  # note that this is not the current state of the arm

    @property
    def gripper(self) -> np.ndarray:
        return self._gripper

    @property
    def led(self) -> np.ndarray:
        return self._led

    @property
    def phi_dot(self) -> np.ndarray:
        # Update phi_dot to current state of the arm
        self._phi_dot = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        return self._phi_dot
