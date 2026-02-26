"""
Ball_XYZ_to_Trajectory.py

Python translation of the MATLAB live script Ball_XYZ_to_Trajectory.mlx.

It simulates noisy (x,y,z) measurements of a moving ball, continuously fits a
trajectory model over a sliding time window, and predicts a future position.
The final fitted model is plotted and the polynomial equations are printed.

Dependencies:
  - numpy
  - matplotlib

Run:
  python Ball_XYZ_to_Trajectory.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def main() -> None:
    # === Parameters (same as MATLAB) ===
    dt = 0.033
    window_size = 20
    noise_level = 0.2

    t_list: list[float] = []
    pos_list: list[Tuple[float, float, float]] = []
    traj: Optional[Trajectory] = None

    # === Simulate measurements and update model online ===
    for k in range(1, 101):
        tk = k * dt
        x_true = 2 * tk
        y_true = 1.5 * tk
        z_true = 1 - 4.9 * tk**2

        x_messy = x_true + noise_level * np.random.randn()
        y_messy = y_true + noise_level * np.random.randn()
        z_messy = z_true + noise_level * np.random.randn()

        t_list.append(tk)
        pos_list.append((x_messy, y_messy, z_messy))

        if len(t_list) > 5:
            t_arr = np.array(t_list, dtype=float)
            pos_arr = np.array(pos_list, dtype=float)

            traj = update_trajectory(t_arr, pos_arr, window_size)

            # Predict 0.3 sec into future (kept for parity with MATLAB)
            t_predict = tk + 0.3
            future_pos = traj.pos(t_predict)[:, 0]  # (3,)
            _ = future_pos  # use if needed

    if traj is None:
        raise RuntimeError("Trajectory was never estimated (not enough samples).")

    # === Plot measured samples and final fitted model ===
    t = np.array(t_list, dtype=float)
    pos = np.array(pos_list, dtype=float)

    t_model = np.linspace(t[0], t[-1] + 0.3, 200)
    model_vals = traj.pos(t_model).T  # (N,3)

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(t, pos[:, 0], "o", label="Measured")
    axes[0].plot(t_model, model_vals[:, 0], linewidth=2, label="Final Model")
    axes[0].set_ylabel("x (m)")
    axes[0].legend()

    axes[1].plot(t, pos[:, 1], "o", label="Measured")
    axes[1].plot(t_model, model_vals[:, 1], linewidth=2, label="Final Model")
    axes[1].set_ylabel("y (m)")
    axes[1].legend()

    axes[2].plot(t, pos[:, 2], "o", label="Measured")
    axes[2].plot(t_model, model_vals[:, 2], linewidth=2, label="Final Model")
    axes[2].set_ylabel("z (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    fig.suptitle("Ball XYZ measurements and fitted trajectory")
    plt.tight_layout()
    plt.show()

    # === Print polynomial equations (rounded like MATLAB) ===
    px = np.round(traj.px.astype(float), 4)
    py = np.round(traj.py.astype(float), 4)
    pz = np.round(traj.pz.astype(float), 4)
    t0 = traj.t0

    print(f"Polynomial equations for the trajectory at t = {t[-1]:.6f}:")
    print(f"x(t) = {px[0]}*(t-t0) + {px[1]}")
    print(f"y(t) = {py[0]}*(t-t0) + {py[1]}")
    print(f"z(t) = {pz[0]}*(t-t0)^2 + {pz[1]}*(t-t0) + {pz[2]}")
    print(f"t0 = {t0}")


if __name__ == "__main__":
    main()