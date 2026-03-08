from re import L
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

    # Updates the phi_cmd based on the camera's output. For now, just returns a dummy command.
    def capture_and_process(self) -> tuple[Optional[np.ndarray], Optional[np.float64], Optional[np.ndarray]]:
        input = self.capture_image()
        # XYZ = self.image_processing(input)
        
        XYZ = np.random.uniform(
            low=np.array([-0.50, -0.10, -0.55], dtype=np.float64),
            high=np.array([-0.40, 0.10, -0.45], dtype=np.float64),
        )
        
        gripper_Cmd = None
        led_Cmd = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        
        return XYZ, gripper_Cmd, led_Cmd
