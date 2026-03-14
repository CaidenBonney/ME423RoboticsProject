from dataclasses import dataclass
import numpy as np


@dataclass
class Trajectory:
    """Holds polynomial coefficients and callable position/velocity functions."""
    def __init__(self) -> None:
        self.px = np.array([0, 0]) # degree 1 polynomial coeffs for x(t_shift)
        self.py = np.array([0, 0]) # degree 1 polynomial coeffs for y(t_shift)
        self.pz = np.array([0, 0, 0]) # degree 2 polynomial coeffs for z(t_shift)
        self.t0 = 0 # time origin for numerical stability
        self.t = np.array([]) 
        self.pos = np.array([]) 

    def predict_pos(self, tt: float | np.ndarray) -> np.ndarray:
        """Position at time tt (seconds). Returns (3,) for scalar tt or (3,N) for array."""
        tt = np.asarray(tt)
        t_shift = tt - self.t0
        # print("self.px: ", self.px)
        # print("self.py: ", self.py)
        # print("self.pz: ", self.pz)
        x = np.polyval(self.px, t_shift)
        y = np.polyval(self.py, t_shift)
        z = np.polyval(self.pz, t_shift)
        return np.vstack([x, y, z])

    def predict_vel(self, tt: float | np.ndarray) -> np.ndarray:
        """Velocity at time tt (seconds). Returns (3,) for scalar tt or (3,N) for array."""
        tt = np.asarray(tt)
        t_shift = tt - self.t0
        vx = np.polyval(np.polyder(self.px), t_shift)
        vy = np.polyval(np.polyder(self.py), t_shift)
        vz = np.polyval(np.polyder(self.pz), t_shift)
        return np.vstack([vx, vy, vz])


    def update_trajectory(self, t: np.ndarray, pos: np.ndarray, window_size: int) -> None:
        t = np.asarray(t).reshape(-1)
        pos = np.asarray(pos).reshape(-1, 3)

        # Append new samples
        if self.t.size == 0:
            self.t = t.copy()
            self.pos = pos.copy()
        else:
            self.t = np.concatenate([self.t, t], axis=0)
            self.pos = np.concatenate([self.pos, pos], axis=0)

        # Keep only the most recent window
        if self.t.size > window_size:
            self.t = self.t[-window_size:]
            self.pos = self.pos[-window_size:, :]

        # Shift time for numerical stability
        self.t0 = float(self.t[0])
        t_shift = self.t - self.t0

        # Fit only when enough points exist
        if self.t.size >= 2:
            self.px = np.polyfit(t_shift, self.pos[:, 0], 1)
            self.py = np.polyfit(t_shift, self.pos[:, 1], 1)
        else:
            self.px = np.array([0.0, self.pos[0, 0]])
            self.py = np.array([0.0, self.pos[0, 1]])

        if self.t.size >= 3:
            self.pz = np.polyfit(t_shift, self.pos[:, 2], 2)
        elif self.t.size >= 1:
            self.pz = np.array([0.0, 0.0, self.pos[0, 2]])