from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from intercept_utils import G, weighted_polyfit


@dataclass
class Trajectory:
    """
    Time units are milliseconds.
    x(t_ms) = px[0] * t + px[1]
    y(t_ms) = py[0] * t + py[1]
    z(t_ms) = pz[0] * t^2 + pz[1] * t + pz[2]
    """

    def __init__(self) -> None:
        self.px = np.array([0.0, 0.0], dtype=np.float64)
        self.py = np.array([0.0, 0.0], dtype=np.float64)
        self.pz = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.t0 = 0.0
        self.t = np.array([], dtype=np.float64)
        self.pos = np.empty((0, 3), dtype=np.float64)

    def reset(self) -> None:
        self.__init__()

    def predict_pos(self, tt: float | np.ndarray) -> np.ndarray:
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
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)

        good = np.isfinite(t)
        good = good & np.all(np.isfinite(pos), axis=1)
        t = t[good]
        pos = pos[good]

        if t.size == 0:
            return

        if self.t.size == 0:
            self.t = t.copy()
            self.pos = pos.copy()
        else:
            self.t = np.concatenate([self.t, t], axis=0)
            self.pos = np.concatenate([self.pos, pos], axis=0)

        if self.t.size > window_size:
            self.t = self.t[-window_size:]
            self.pos = self.pos[-window_size:, :]

        self.t0 = float(self.t[0])
        t_shift = self.t - self.t0

        self._fit_polynomials(t_shift)

    def _fit_polynomials(self, t_shift_ms: np.ndarray) -> None:
        n = t_shift_ms.size
        if n == 0:
            return

        age = t_shift_ms[-1] - t_shift_ms
        tau = max(float(t_shift_ms[-1]), 1.0)
        w = np.exp(-1.2 * age / tau)

        if n >= 2:
            self.px = weighted_polyfit(t_shift_ms, self.pos[:, 0], deg=1, w=w)
            self.py = weighted_polyfit(t_shift_ms, self.pos[:, 1], deg=1, w=w)
        else:
            self.px = np.array([0.0, self.pos[0, 0]], dtype=np.float64)
            self.py = np.array([0.0, self.pos[0, 1]], dtype=np.float64)

        if n >= 3:
            self.pz = self._fit_ballisticish_quadratic_ms(t_shift_ms, self.pos[:, 2], w=w)
        elif n == 2:
            p_lin = weighted_polyfit(t_shift_ms, self.pos[:, 2], deg=1, w=w)
            self.pz = np.array([0.0, p_lin[0], p_lin[1]], dtype=np.float64)
        else:
            self.pz = np.array([0.0, 0.0, self.pos[0, 2]], dtype=np.float64)

    @staticmethod
    def _fit_ballisticish_quadratic_ms(t_shift_ms: np.ndarray, z: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        q = weighted_polyfit(t_shift_ms, z, deg=2, w=w)
        q = np.asarray(q, dtype=np.float64)

        # ballistic curvature in ms units
        ballistic_a_ms = -0.5 * G / 1_000_000.0

        # allow weak curvature instead of forcing strong downward curvature
        # this helps when there are only a few points early in the flight
        q[0] = min(q[0], -0.15 * G / 1_000_000.0)

        # if fit becomes numerically tiny, bias it slightly downward
        if abs(q[0]) < abs(ballistic_a_ms) * 0.05:
            q[0] = -0.1 * G / 1_000_000.0

        return q