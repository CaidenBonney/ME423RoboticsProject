from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from intercept_utils import G, weighted_polyfit


@dataclass
class Trajectory:
    """
    Holds polynomial coefficients and callable position/velocity functions.

    Time units in this class are milliseconds, because that is what your
    camera/arm pipeline currently uses.
    """

    def __init__(self) -> None:
        # x(t_shift_ms) = px[0] * t + px[1]
        # y(t_shift_ms) = py[0] * t + py[1]
        # z(t_shift_ms) = pz[0] * t^2 + pz[1] * t + pz[2]
        self.px = np.array([0.0, 0.0], dtype=np.float64)
        self.py = np.array([0.0, 0.0], dtype=np.float64)
        self.pz = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.t0 = 0.0  # scalar milliseconds
        self.t = np.array([], dtype=np.float64)  # timestamps in milliseconds
        self.pos = np.empty((0, 3), dtype=np.float64)

        self.points_since_update = 0
        self.pz_buffer: list[np.ndarray] = []
        self.pz_buffer_size = 1  # keep immediate updates for responsive interception

    def reset(self) -> None:
        self.__init__()

    def predict_pos(self, tt: float | np.ndarray) -> np.ndarray:
        """
        Position at time tt (milliseconds).
        Returns shape (3,1) for scalar input and (3,N) for array input.
        """
        tt_arr = np.asarray(tt, dtype=np.float64)
        if self.t.size == 0:
            if tt_arr.ndim == 0:
                return np.zeros((3, 1), dtype=np.float64)
            return np.zeros((3, tt_arr.size), dtype=np.float64)

        t_shift = tt_arr - self.t0
        x = np.polyval(self.px, t_shift)
        y = np.polyval(self.py, t_shift)
        z = np.polyval(self.pz, t_shift)

        out = np.vstack([x, y, z]).astype(np.float64)
        if tt_arr.ndim == 0:
            return out.reshape(3, 1)
        return out

    def predict_vel(self, tt: float | np.ndarray) -> np.ndarray:
        """
        Velocity at time tt (milliseconds).
        Returns shape (3,1) for scalar input and (3,N) for array input.
        Velocity units are meters per millisecond.
        """
        tt_arr = np.asarray(tt, dtype=np.float64)
        if self.t.size == 0:
            if tt_arr.ndim == 0:
                return np.zeros((3, 1), dtype=np.float64)
            return np.zeros((3, tt_arr.size), dtype=np.float64)

        t_shift = tt_arr - self.t0
        vx = np.polyval(np.polyder(self.px), t_shift)
        vy = np.polyval(np.polyder(self.py), t_shift)
        vz = np.polyval(np.polyder(self.pz), t_shift)

        out = np.vstack([vx, vy, vz]).astype(np.float64)
        if tt_arr.ndim == 0:
            return out.reshape(3, 1)
        return out

    def update_trajectory(self, t: np.ndarray, pos: np.ndarray, window_size: int, update_freq: int = 0) -> None:
        """
        Add one or more samples and refit.
        Inputs:
            t   : timestamp(s) in milliseconds
            pos : xyz sample(s) in meters, shape (3,) or (N,3)
        """
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)

        # Filter bad rows early
        good = np.all(np.isfinite(pos), axis=1) & np.isfinite(t)
        t = t[good]
        pos = pos[good]

        if t.size == 0:
            return

        # Append
        if self.t.size == 0:
            self.t = t.copy()
            self.pos = pos.copy()
        else:
            self.t = np.concatenate([self.t, t], axis=0)
            self.pos = np.concatenate([self.pos, pos], axis=0)

        # Keep only recent window
        if self.t.size > window_size:
            self.t = self.t[-window_size:]
            self.pos = self.pos[-window_size:, :]

        self.t0 = float(self.t[0])
        t_shift = self.t - self.t0  # milliseconds

        self._fit_polynomials(t_shift)

    def _fit_polynomials(self, t_shift_ms: np.ndarray) -> None:
        """
        Robust-ish weighted fit:
        - x/y: linear
        - z: quadratic with concave-down / gravity-consistent curvature
        """
        n = t_shift_ms.size
        if n == 0:
            return

        # Weight recent samples more heavily
        age = t_shift_ms[-1] - t_shift_ms
        tau = max(float(t_shift_ms[-1]), 1.0)
        w = np.exp(-2.0 * age / tau)

        # x/y fits
        if n >= 2:
            self.px = weighted_polyfit(t_shift_ms, self.pos[:, 0], deg=1, w=w)
            self.py = weighted_polyfit(t_shift_ms, self.pos[:, 1], deg=1, w=w)
        else:
            self.px = np.array([0.0, self.pos[0, 0]], dtype=np.float64)
            self.py = np.array([0.0, self.pos[0, 1]], dtype=np.float64)

        # z fit
        if n >= 3:
            new_pz = self._fit_concave_down_quadratic_ms(t_shift_ms, self.pos[:, 2], w=w)
            self._update_pz_batch(new_pz)
        elif n >= 2:
            # fall back to linear z packed as quadratic
            pz_lin = weighted_polyfit(t_shift_ms, self.pos[:, 2], deg=1, w=w)
            self.pz = np.array([0.0, pz_lin[0], pz_lin[1]], dtype=np.float64)
        else:
            self.pz = np.array([0.0, 0.0, self.pos[0, 2]], dtype=np.float64)

    @staticmethod
    def _fit_concave_down_quadratic_ms(t_shift_ms: np.ndarray, z: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """
        Fit z(t_ms) = a t_ms^2 + b t_ms + c.
        Enforce concave-down with at least ballistic gravity curvature in ms-units.

        In seconds, ideal ballistic curvature is -g/2.
        In milliseconds, that becomes:
            a = -(g/2) / (1000^2)
        """
        q = weighted_polyfit(t_shift_ms, z, deg=2, w=w)
        ballistic_a_ms = -0.5 * G / 1_000_000.0  # meters / ms^2
        q = np.asarray(q, dtype=np.float64)
        q[0] = min(q[0], ballistic_a_ms)
        return q

    def _update_pz_batch(self, new_pz: np.ndarray) -> None:
        self.pz_buffer.append(np.asarray(new_pz, dtype=np.float64).copy())
        if len(self.pz_buffer) >= self.pz_buffer_size:
            self.pz = np.mean(np.stack(self.pz_buffer, axis=0), axis=0)
            self.pz_buffer.clear()