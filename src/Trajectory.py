from dataclasses import dataclass
import numpy as np


@dataclass
class Trajectory:
    """Holds polynomial coefficients and callable position/velocity functions."""

    px: np.ndarray  # degree 1 polynomial coeffs for x(t_shift)
    py: np.ndarray  # degree 1 polynomial coeffs for y(t_shift)
    pz: np.ndarray  # degree 2 polynomial coeffs for z(t_shift)
    t0: float       # time origin for numerical stability

    def pos(self, tt: float | np.ndarray) -> np.ndarray:
        """Position at time tt (seconds). Returns (3,) for scalar tt or (3,N) for array."""
        tt = np.asarray(tt)
        t_shift = tt - self.t0
        x = np.polyval(self.px, t_shift)
        y = np.polyval(self.py, t_shift)
        z = np.polyval(self.pz, t_shift)
        return np.vstack([x, y, z])

    def vel(self, tt: float | np.ndarray) -> np.ndarray:
        """Velocity at time tt (seconds). Returns (3,) for scalar tt or (3,N) for array."""
        tt = np.asarray(tt)
        t_shift = tt - self.t0
        vx = np.polyval(np.polyder(self.px), t_shift)
        vy = np.polyval(np.polyder(self.py), t_shift)
        vz = np.polyval(np.polyder(self.pz), t_shift)
        return np.vstack([vx, vy, vz])


def update_trajectory(t: np.ndarray, pos: np.ndarray, window_size: int) -> Trajectory:
    """
    Fit a trajectory model on the most recent window of samples.

    MATLAB original:
      - keep only last window_size samples
      - shift time by t0 = first time in window
      - fit:
          x(t_shift) = linear polyfit degree 1
          y(t_shift) = linear polyfit degree 1
          z(t_shift) = quadratic polyfit degree 2
    """
    t = np.asarray(t).reshape(-1)
    pos = np.asarray(pos)

    if t.size > window_size:
        t = t[-window_size:]
        pos = pos[-window_size:, :]

    t0 = float(t[0])
    t_shift = t - t0

    px = np.polyfit(t_shift, pos[:, 0], 1)
    py = np.polyfit(t_shift, pos[:, 1], 1)
    pz = np.polyfit(t_shift, pos[:, 2], 2)

    return Trajectory(px=px, py=py, pz=pz, t0=t0)
