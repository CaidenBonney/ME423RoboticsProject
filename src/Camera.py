import time
import numpy as np


class Camera:
    def __init__(self):
        self.startTime = time.time()
        self.sampleRate = 30
        self.sampleTime = 1 / self.sampleRate

        self.cam_setup()

    def elapsed_time(self):
        return time.time() - self.startTime

    def cam_setup(self):
        # TODO: add camera initialization (device open, stream config, etc.).
        self.cam_calibration()

    def cam_calibration(self):
        # TODO: add camera calibration logic (make rerunable).
        pass

    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self):
        # TODO: define camera output to phi_cmd mapping. (maybe done in trajectory planning instead?)
        phi_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        gripper_cmd = 0.0
        led_cmd = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        return phi_cmd, gripper_cmd, led_cmd
