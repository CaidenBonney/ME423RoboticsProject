import time
from typing import Optional
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

    def capture_image(self) -> np.ndarray:
        # TODO: add camera image capture logic.
        pass

    def image_processing(self, image: np.ndarray) -> np.ndarray:
        # TODO: convert image to XYZ
        pass

    def XYZ_to_phi_cmd(
        self, XYZ: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.float64], Optional[np.ndarray]]:

        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]

        
        
        phi_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        gripper_cmd = np.float64(0.0)
        led_cmd = None

        return phi_cmd, gripper_cmd, led_cmd

    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self) -> tuple[Optional[np.ndarray], Optional[np.float64], Optional[np.ndarray]]:
        input = self.capture_image()
        XYZ = self.image_processing(input)
        return self.XYZ_to_phi_cmd(XYZ)
