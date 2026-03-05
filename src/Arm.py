# Special QArm library imports
from pal.products.qarm import QArm  # pyright: ignore[reportMissingImports]
from hal.products.qarm import QArmUtilities  # pyright: ignore[reportMissingImports]

# Standard library imports
import time
import numpy as np


class Arm:
    def __init__(self):
        # Internal variables from basic position mode py files
        self.startTime = time.time()
        self.sampleRate = 200
        self.sampleTime = 1 / self.sampleRate
        self.myArm = QArm(hardware=1)
        self.myArmUtilities = QArmUtilities()
        # print("Sample Rate is ", self.sampleRate, " Hz. Simulation will run until you type Ctrl+C to exit.")

        # Internal variables for current state of the arm
        self._phi = [0, 0, 0, 0]  # [rad] joint angles for 4 joints in order: base, shoulder, elbow, wrist
        self._phi_dot = [0, 0, 0, 0]  # [rad/s]
        self._position = [0, 0, 0]  # [m]
        self._gripper = 0.0  # [0 = open, 1 = closed]
        self._led = np.array([0, 0, 0], dtype=np.float64)  # [R, G, B] values as floats from 0 to 1

    def elapsed_time(self):
        return time.time() - self.startTime

    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None):
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
    def limit_check(self, phi_cmd):
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
    def workspace_check(self, phi_cmd):
        # TODO: Update workspace limits to real values and test phi_cmd to make sure it won't run into the table

        # Base parrallel to table so impossible to run into table with any base angle
        if self._phi[1] < -np.radians(85) or self._phi[1] > np.radians(85):
            raise ValueError("Shoulder Workspace limit reached. Arm moved to home position.")
        elif self._phi[2] < -np.radians(95) or self._phi[2] > np.radians(75):
            raise ValueError("Elbow Workspace limit reached. Arm moved to home position.")
        elif self._phi[3] < -np.radians(160) or self._phi[3] > np.radians(160):
            raise ValueError("Wrist Workspace limit reached. Arm moved to home position.")

    def home(self):
        self._phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self._led = np.array([1, 0, 0], dtype=np.float64)
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

    @property
    def phi(self):
        # Update phi to current state of the arm
        self._phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)
        return self._phi

    @property
    def position(self):
        # Update phi to current state of the arm and then calculate position with forward kinematics
        self._position = self.myArmUtilities.forward_kinematics(self.phi)  # this will update self._phi as well
        return self._position

    @property
    def gripper(self):
        return self._gripper

    @property
    def led(self):
        return self._led

    @property
    def phi_dot(self):
        # Update phi_dot to current state of the arm
        self._phi_dot = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        return self._phi_dot
