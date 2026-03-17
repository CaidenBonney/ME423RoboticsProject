from dataclasses import dataclass
import numpy as np
import scipy.optimize as sp


@dataclass
class Trajectory:
    """Holds polynomial coefficients and callable position/velocity functions."""

    def __init__(self) -> None:
        self.px = np.array([0, 0])  # degree 1 polynomial coeffs for x(t_shift)
        self.py = np.array([0, 0])  # degree 1 polynomial coeffs for y(t_shift)
        self.pz = np.array([0, 0, 0])  # degree 2 polynomial coeffs for z(t_shift)
        self.t0 = 0  # time origin for numerical stability (milliseconds). Gets set to first point timestamp
        self.t = np.array([])  # timestamps of observed points (milliseconds)
        self.pos = np.array([0, 0, 0]).reshape(3, 1)
        self.points_since_update = 5  # count how many points have been added since the last trajectory update, used to determine when to update the trajectory fit
        self.pz_buffer = []
        self.pz_buffer_size = 3

    def predict_pos(self, tt: float | np.ndarray) -> np.ndarray:
        """Position at time tt (milliseconds). Returns (3,) for scalar tt or (3,N) for array."""
        # print("")
        if np.size(self.t) == 0:
            return np.zeros((3,)) if np.size(tt) > 1 else np.zeros((3, 1))
        tt = np.asarray(tt)
        t_shift = tt - self.t0  # [milliseconds] shift by t0 for numerical stability
        # print("t_shift: ", t_shift)
        # print("self.px: ", self.px)
        # print("self.py: ", self.py)
        # print("self.pz: ", self.pz)
        x = np.polyval(self.px, t_shift)
        y = np.polyval(self.py, t_shift)
        z = np.polyval(self.pz, t_shift)
        # x = self.pos[-1,0]
        # y = self.pos[-1,1]
        # z = self.pos[-1,2]
        return np.vstack([x, y, z])

    def predict_vel(self, tt: float | np.ndarray) -> np.ndarray:
        """Velocity at time tt (milliseconds). Returns (3,) for scalar tt or (3,N) for array."""
        tt = np.asarray(tt)
        t_shift = tt - self.t0  # [milliseconds] shift by t0 for numerical stability
        vx = np.polyval(np.polyder(self.px), t_shift)
        vy = np.polyval(np.polyder(self.py), t_shift)
        vz = np.polyval(np.polyder(self.pz), t_shift)
        return np.vstack([vx, vy, vz])

    def update_trajectory(self, t: np.ndarray, pos: np.ndarray, window_size: int, update_freq: int = 0) -> None:
        # t is the timestamp of the frame in which the point was detected in global time (milliseconds)
        t = np.asarray(t).reshape(-1)
        pos = np.asarray(pos).reshape(-1, 3)

        # Append new samples
        if self.t.size == 0:
            self.t = np.append(self.t, t)
            self.pos = pos.copy()
            self.t0 = t
        else:
            self.t = np.concatenate([self.t, t], axis=0)
            self.pos = np.concatenate([self.pos, pos], axis=0)

        # Keep only the most recent window
        if self.t.size > window_size:
            self.t = self.t[-window_size:]
            self.t0 = self.t[0]  # update t0 to the new oldest timestamp for numerical stability
            self.pos = self.pos[-window_size:, :]

        # print("t0: ", self.t0)
        t_shift = self.t - self.t0  # [milliseconds] shift by t0 for numerical stability
        # print ("t0: ", self.t0,"t_shift: ", t_shift)

        # Fit only when enough points exist
        if self.t.size >= 2:
            if self.points_since_update >= update_freq:
                self.px = np.polyfit(t_shift, self.pos[:, 0], 1)
                self.py = np.polyfit(t_shift, self.pos[:, 1], 1)
                self.points_since_update = 0
            else:
                self.points_since_update += 1
        else:
            self.px = np.array([0.0, self.pos[0, 0]])
            self.py = np.array([0.0, self.pos[0, 1]])

        if self.t.size >= 3:
            # if self.points_since_update >= 3:
                # # self.pz = np.polyfit(t_shift, self.pos[:, 2],
                # self.pz = sp.curve_fit(lambda t, a, b, c: a * t**2 + b * t + c, t_shift, self.pos[:, 2])[
                #     0
                # ]  # , bounds=([0, -np.inf, -np.inf], [-np.inf, np.inf, np.inf]))[0]
                # self.points_since_update = 0
                # print("z fit coeffs: ", self.pz)
                
            if self.points_since_update >= 0:
                new_pz = self._fit_concave_down_quadratic(t_shift, self.pos[:, 2])
                self._update_pz_batch(new_pz)
                self.points_since_update = 0
            else:
                self.points_since_update += 1
        elif self.t.size >= 1:
            self.pz = np.array([0.0, 0.0, self.pos[0, 2]])

    @staticmethod
    def _fit_concave_down_quadratic(t_shift: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Fast least-squares fit of z = a t^2 + b t + c with constraint a <= 0.
        Exact solution without iterative optimization:
            - use quadratic fit if unconstrained a <= 0
            - otherwise best constrained fit is the boundary case a = 0 (a line)
        Returns [a, b, c].
        """
        # Unconstrained quadratic least squares
        X2 = np.column_stack((t_shift * t_shift, t_shift, np.ones_like(t_shift)))
        q, _, _, _ = np.linalg.lstsq(X2, z, rcond=None)
        a, b, c = q

        q[0] = min(q[0], -0.5*9.81/1_000_000)  # enforce a <= 0 constraint
        print(f"Fitted quadratic coeffs: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        return q

    def _update_pz_batch(self, new_pz: np.ndarray) -> None:
        """
        Collect raw fits, then update self.pz to their average and flush buffer at specified size.
        """
        self.pz_buffer.append(new_pz.copy())

        if len(self.pz_buffer) == self.pz_buffer_size:
            self.pz = np.mean(np.stack(self.pz_buffer, axis=0), axis=0)
            self.pz_buffer.clear()
            print("batched z fit coeffs:", self.pz)