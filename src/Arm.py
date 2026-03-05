# Special QArm library imports
from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities

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
        start = self.elapsed_time()
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

        # region: Phi Limit Check
        #   Phi limits (from QArm documentation):
        #       Base: ± 170 deg
        #       Shoulder: ± 85 deg
        #       Elbow: -95 deg / +75 deg
        #       Wrist: ± 160 deg

        # TODO: Note the arm can still run into the table as of now, so we may want to add some
        # workspace limits in the future as well. For now, we just check joint limits and move
        # to home position if they are exceeded.
        if self._phi[0] < -np.radians(170) or self._phi[0] > np.radians(170):
            self.home()
            raise ValueError("Phi limit reached. Arm moved to home position.")
        if self._phi[1] < -np.radians(85) or self._phi[1] > np.radians(85):
            self.home()
            raise ValueError("Phi limit reached. Arm moved to home position.")
        elif self._phi[2] < -np.radians(95) or self._phi[2] > np.radians(75):
            self.home()
            raise ValueError("Phi limit reached. Arm moved to home position.")
        elif self._phi[3] < -np.radians(160) or self._phi[3] > np.radians(160):
            self.home()
            raise ValueError("Phi limit reached. Arm moved to home position.")
        # endregion

        # region: Send Commands to Arm
        try: 
            while self.myArm.status:
                self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

                # Pause/sleep to maintain Rate
                sleep_time = self.sampleTime - (self.elapsed_time() - start) % self.sampleTime
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("User interrupted!")
        # endregion

    def home(self):
        self._phi = np.array([0, 0, 0, 0], dtype=np.float64)
        self._led = np.array([1, 0, 0], dtype=np.float64)
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)

    @property
    def phi(self):
        # phi to current state of the arm
        self._phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)
        return self._phi

    @property
    def position(self):
        self._position = self.myArmUtilities.forward_kinematics(self._phi)
        return self._position

    @property
    def gripper(self):
        return self._gripper

    @property
    def led(self):
        return self._led

    @property
    def phi_dot(self):
        # update phi_dot to current state of the arm
        self._phi_dot = np.asarray(self.myArm.measJointSpeed[0:4], dtype=np.float64)
        return self._phi_dot
