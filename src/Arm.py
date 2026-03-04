# Special QArm library imports
from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities

# Standard library imports
import time
import numpy as np


class Arm:
    def __init__(self):
        # Variables from basic position mode py files
        self.startTime = time.time()
        self.sampleRate = 200
        self.sampleTime = 1 / self.sampleRate
        self.myArm = QArm(hardware=1)
        self.myArmUtilities = QArmUtilities()
        # print("Sample Rate is ", self.sampleRate, " Hz. Simulation will run until you type Ctrl+C to exit.")

        # Variables for current state of the arm
        self.phi = [0, 0, 0, 0]  # [rad]
        self.phi_dot = [0, 0, 0, 0]  # [rad/s]
        self.position = [0, 0, 0]  # [m]
        self.gripper = 0  # [0 = open, 1 = closed]
        self.led = np.array([0, 0, 0], dtype=np.float64)  # [R, G, B] values as floats from 0 to 1

    def elapsed_time(self):
        return time.time() - self.startTime

    def move(self, joints, gripper, led):
        start = self.elapsed_time()


        self.led = np.array([0, 1, 1], dtype=np.float64)
        result = self.myArmUtilities.take_user_input_joint_space()
        phiCmd = result[0:4]
        gripCmd = result[4]
        print(f"Total time elapsed: {int(self.elapsed_time())} seconds.")
        
        self.myArm.read_write_std(phiCMD=phiCmd, grpCMD=gripCmd, baseLED=self.led)
        self.phi_dot = self.myArm.measJointSpeed

        # Pause/sleep to maintain Rate
        time.sleep(self.sampleTime - (self.elapsed_time() - start) % self.sampleTime)

        self.phi = joints
        self.gripper = gripper
        self.led = led

    def get_joints(self):
        return self.phi

    def get_position(self):
        self.position = self.myArmUtilities.forward_kinematics(self.phi)
        return self.position

    def get_gripper(self):
        return self.gripper

    def get_led(self):
        return self.led
