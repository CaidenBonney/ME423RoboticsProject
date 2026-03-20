import importlib
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# from Trajectory import Trajectory
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Trajectory.py"))

# Load the module dynamically
spec = importlib.util.spec_from_file_location("Trajectory", module_path)
TRAJECTORY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TRAJECTORY)

def main() -> None:
    # === Parameters (same as MATLAB) ===
    dt = 0.033
    window_size = 1000
    noise_level = 0.2

    t_list: list[float] = []
    pos_list: list[Tuple[float, float, float]] = []
    traj = TRAJECTORY.Trajectory()

    # === Simulate measurements and update model online ===
    for k in range(1, 101):
        tk = k * dt
        # Input initial velocities/positions here
        x_true = 2 * tk
        y_true = 1.5 * tk
        z_true = -2 + 10*tk - 4.9 * tk**2

        x_messy = x_true + noise_level * np.random.randn()
        y_messy = y_true + noise_level * np.random.randn()
        z_messy = z_true + noise_level * np.random.randn()

        t_list.append(tk)
        pos_list.append((x_messy, y_messy, z_messy))

        if len(t_list) > 5:
            t_arr = np.array(t_list, dtype=float)
            pos_arr = np.array(pos_list, dtype=float)

            traj.update_trajectory(t_arr, pos_arr, window_size)

            # Predict 0.3 sec into future (kept for parity with MATLAB)
            t_predict = tk + 0.3
            future_pos = traj.predict_pos(t_predict)[:, 0]  # (3,)
            _ = future_pos  # use if needed

    if traj is None:
        raise RuntimeError("Trajectory was never estimated (not enough samples).")

    t = np.array(t_list, dtype=float)
    pos = np.array(pos_list, dtype=float)

    t_model = np.linspace(t[0], t[-1] + 0.3, 200)
    model_vals = traj.predict_pos(t_model).T  # (N,3)

    px = traj.px
    py = traj.py
    pz = traj.pz
    t0 = traj.t0
    

    # === Plot measured samples and final fitted model ===
    '''
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
    '''

    # === Calculating intercept time (t_p) and intercept position (p_p) ===
    # Coefficients to solve z(t) = z_p
    t_p = None
    z_p = 0.49
    print(traj.pz[0])
    print(traj.pz[1])
    print(traj.pz[2])
    coeffs = np.array([traj.pz[0], traj.pz[1], traj.pz[2] - z_p])
    t_roots = np.roots(coeffs)
    print(t_roots)
    print(traj.t0)
    t_roots = t_roots[np.isreal(t_roots)].real

    # Verify that the ball is on the way down when t_p is selected
    for r in t_roots: 
        t_world = r + traj.t0
        vz = traj.predict_vel(t_world)[2]  # vertical velocity
        if vz < 0:   # descending
            t_p = t_world
            break
    
    if t_p is None:
        print("No valid intercept found")
        return

    # Set the intercept location

    p_p = traj.predict_pos(np.array([t_p]))[:, 0]

    x_p, y_p, z_p_actual = p_p

    print(f"Predicted intercept position at t_p = {t_p:.4f}s:")
    print(f"x = {x_p:.4f}, y = {y_p:.4f}, z = {z_p_actual:.4f}")

    # === Verification plot of z_p ===

    plt.figure(2)
    plt.plot(t, pos[:, 2], "o", label="Measured")
    plt.xlim(0, 2.5)
    plt.ylim(-2, 7)
    plt.plot(t_model, model_vals[:, 2], linewidth=2, label="Final Model")
    plt.plot(t_p, z_p, "ro", markersize=10, label="Commanded Intercept")
    plt.ylabel("z (m)")
    plt.xlabel("Time (s)")
    plt.axhline(y=z_p, color='r', linestyle='-')
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    main()
