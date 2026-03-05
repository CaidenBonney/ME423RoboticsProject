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
        self._phi = [0, 0, 0, 0]  # [rad]
        self._phi_dot = [0, 0, 0, 0]  # [rad/s]
        self._position = [0, 0, 0]  # [m]
        self._gripper = 0.0  # [0 = open, 1 = closed]
        self._led = np.array([0, 0, 0], dtype=np.float64)  # [R, G, B] values as floats from 0 to 1

    def elapsed_time(self):
        return time.time() - self.startTime

    def move(self, phi_Cmd, gripper_Cmd=None, led_Cmd=None):
        # region: Process Inputs:
        # Update phi to the commanded values
        self._phi = np.asarray(phi_Cmd, dtype=np.float64)
        if phi_Cmd.shape != (4,):
            raise ValueError("phi_Cmd must be an iterable of 4 joint angles [rad].")

        # Optional gripper command
        if gripper_Cmd is not None:
            if gripper_Cmd < 0 or gripper_Cmd > 1:
                raise ValueError("gripper_Cmd must be a value between 0 and 1.")
            self._gripper = np.asarray(gripper_Cmd, dtype=np.float64)

        # Optional led command
        if led_Cmd is not None:
            led_Cmd = np.asarray(led_Cmd, dtype=np.float64)
            if led_Cmd.shape != (3,):
                raise ValueError("led_Cmd must be an iterable of 3 values [R, G, B].")
            self._led = led_Cmd
        # endregion
        
        # region: Phi Limit Check
        if self._phi[0] < -np.pi:
            self._phi[0] = -np.pi
        # endregion
        
        start = self.elapsed_time()
        
        # region: Send Commands to Arm
        self.myArm.read_write_std(phiCMD=self._phi, grpCMD=self._gripper, baseLED=self._led)
        # endregion

        # Pause/sleep to maintain Rate
        sleep_time = self.sampleTime - (self.elapsed_time() - start) % self.sampleTime
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Keep state aligned with measured feedback when available.
        self._phi = np.asarray(self.myArm.measJointPosition[0:4], dtype=np.float64)

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
