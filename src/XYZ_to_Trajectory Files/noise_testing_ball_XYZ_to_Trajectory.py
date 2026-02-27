from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ball_XYZ_to_Trajectory import Trajectory
from ball_XYZ_to_Trajectory import update_trajectory


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
