import time
import numpy as np
from numpy.random import randint


class Camera:
    def __init__(self) -> None:
        self.startTime = time.time()
        self.sampleRate = 30
        self.sampleTime = 1 / self.sampleRate

        self.cam_setup()

    def elapsed_time(self) -> float:
        return time.time() - self.startTime

    def cam_setup(self) -> None:
        # TODO: add camera initialization (device open, stream config, etc.).
        self.cam_calibration()

    def cam_calibration(self) -> None:
        # TODO: add camera calibration logic (make rerunable).
        pass

    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self) -> tuple[np.ndarray, np.float64, np.ndarray]:
        # TODO: define camera output to phi_cmd mapping. (maybe done in trajectory planning instead?)

        tol = 0.2

        phi_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) + np.clip(np.random.randn(4) * 0.1, -tol, tol)
        gripper_cmd = np.float64(0.0)
        led_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64) + np.clip(np.random.randn(3) + 0.5, 0, 1)

        return phi_cmd, gripper_cmd, led_cmd
