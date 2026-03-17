from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


G = 9.81  # m/s^2


@dataclass
class KalmanInterceptResult:
    valid: bool
    t_hit_ms: Optional[float] = None
    xyz_hit: Optional[np.ndarray] = None
    xy_hit: Optional[np.ndarray] = None
    confidence: float = 0.0
    reason: str = ""


def solve_quadratic_real(a: float, b: float, c: float) -> np.ndarray:
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return np.array([], dtype=np.float64)
        return np.array([-c / b], dtype=np.float64)

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return np.array([], dtype=np.float64)

    s = np.sqrt(disc)
    return np.array([(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)], dtype=np.float64)


def choose_future_root(roots: np.ndarray, min_future: float) -> Optional[float]:
    roots = np.asarray(roots, dtype=np.float64)
    roots = roots[np.isfinite(roots)]
    roots = roots[roots >= min_future]
    if roots.size == 0:
        return None
    return float(np.min(roots))


class BallisticKalmanFilter:
    """
    State:
        [x, y, z, vx, vy, vz]^T
    Units:
        position in meters
        velocity in m/s
        timestamps in milliseconds externally
    """

    def __init__(self) -> None:
        self.x = np.zeros((6, 1), dtype=np.float64)

        # position fairly certain, velocity very uncertain at startup
        self.P = np.diag([0.01, 0.01, 0.01, 4.0, 4.0, 6.0]).astype(np.float64)

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # trust measurements, but not absurdly strongly
        self.R = np.diag([0.004**2, 0.004**2, 0.007**2]).astype(np.float64)

        self.Q_base = np.diag([2e-4, 2e-4, 3e-4, 0.35, 0.35, 0.45]).astype(np.float64)

        self.initialized = False
        self.last_t_ms: Optional[float] = None

        self.history_t_ms: list[float] = []
        self.history_xyz: list[np.ndarray] = []

        self.bootstrap_ready = False
        self.bootstrap_min_points = 4
        self.bootstrap_max_points = 8

    def reset(self) -> None:
        self.__init__()

    def _F_B(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt_s
        F[1, 4] = dt_s
        F[2, 5] = dt_s

        B = np.zeros((6, 1), dtype=np.float64)
        B[2, 0] = 0.5 * dt_s * dt_s
        B[5, 0] = dt_s
        return F, B

    def _Q(self, dt_s: float) -> np.ndarray:
        scale = max(dt_s, 1e-3)
        Q = self.Q_base.copy()
        Q[0, 0] *= scale
        Q[1, 1] *= scale
        Q[2, 2] *= scale
        Q[3, 3] *= scale
        Q[4, 4] *= scale
        Q[5, 5] *= scale
        return Q

    def _append_history(self, timestamp_ms: float, xyz: np.ndarray) -> None:
        self.history_t_ms.append(float(timestamp_ms))
        self.history_xyz.append(np.asarray(xyz, dtype=np.float64).reshape(3).copy())

        if len(self.history_t_ms) > 80:
            self.history_t_ms = self.history_t_ms[-80:]
            self.history_xyz = self.history_xyz[-80:]

    def _bootstrap_from_history(self) -> bool:
        """
        Build physically meaningful initial state from first few points.

        x(t) = x0 + vx*t
        y(t) = y0 + vy*t
        z(t) = z0 + vz*t - 0.5*g*t^2

        Rearranged:
        z(t) + 0.5*g*t^2 = z0 + vz*t
        """
        if len(self.history_t_ms) < self.bootstrap_min_points:
            return False

        t_ms = np.asarray(self.history_t_ms[-self.bootstrap_max_points:], dtype=np.float64)
        xyz = np.asarray(self.history_xyz[-self.bootstrap_max_points:], dtype=np.float64)

        t0_ms = t_ms[0]
        t_s = (t_ms - t0_ms) / 1000.0

        if np.max(t_s) < 1e-3:
            return False

        # Fit x and y linearly
        px = np.polyfit(t_s, xyz[:, 0], 1)
        py = np.polyfit(t_s, xyz[:, 1], 1)

        # Fit ballistic z using known gravity
        z_lin = xyz[:, 2] + 0.5 * G * t_s * t_s
        pz_lin = np.polyfit(t_s, z_lin, 1)

        vx = float(px[0])
        x0 = float(px[1])

        vy = float(py[0])
        y0 = float(py[1])

        vz = float(pz_lin[0])
        z0 = float(pz_lin[1])

        # Initialize current state at the most recent time in the bootstrap window
        t_now_s = float(t_s[-1])

        x_now = x0 + vx * t_now_s
        y_now = y0 + vy * t_now_s
        z_now = z0 + vz * t_now_s - 0.5 * G * t_now_s * t_now_s
        vz_now = vz - G * t_now_s

        self.x[:, 0] = np.array([x_now, y_now, z_now, vx, vy, vz_now], dtype=np.float64)

        # Larger velocity uncertainty than position uncertainty
        self.P = np.diag([0.003, 0.003, 0.004, 1.5, 1.5, 2.0]).astype(np.float64)

        self.initialized = True
        self.bootstrap_ready = True
        self.last_t_ms = float(t_ms[-1])
        return True

    def update(self, timestamp_ms: float, xyz: np.ndarray) -> None:
        z = np.asarray(xyz, dtype=np.float64).reshape(3, 1)

        if not np.all(np.isfinite(z)):
            return

        self._append_history(float(timestamp_ms), z[:, 0])

        # Bootstrap first
        if not self.initialized:
            ok = self._bootstrap_from_history()
            if not ok:
                return

            # If the current measurement is newer than the bootstrap state time,
            # continue into standard update below
            if float(timestamp_ms) == float(self.last_t_ms):
                return

        dt_s = max((float(timestamp_ms) - float(self.last_t_ms)) / 1000.0, 1e-4)
        self.last_t_ms = float(timestamp_ms)

        # Predict
        F, B = self._F_B(dt_s)
        u = np.array([[-G]], dtype=np.float64)
        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + self._Q(dt_s)

        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ self.H) @ self.P

    def predict_state_at_dt(self, dt_s: float) -> np.ndarray:
        if not self.initialized:
            return np.full(6, np.nan, dtype=np.float64)

        x0, y0, z0, vx, vy, vz = self.x[:, 0]

        xf = x0 + vx * dt_s
        yf = y0 + vy * dt_s
        zf = z0 + vz * dt_s - 0.5 * G * dt_s * dt_s
        vxf = vx
        vyf = vy
        vzf = vz - G * dt_s

        return np.array([xf, yf, zf, vxf, vyf, vzf], dtype=np.float64)

    def predict_plane_intercept(
        self,
        z_catch: float,
        min_lead_ms: float,
        max_horizon_ms: float,
    ) -> KalmanInterceptResult:
        if not self.initialized or self.last_t_ms is None:
            return KalmanInterceptResult(valid=False, reason="filter not initialized")

        x0, y0, z0, vx, vy, vz = self.x[:, 0]

        roots_s = solve_quadratic_real(-0.5 * G, vz, z0 - z_catch)
        t_hit_s = choose_future_root(roots_s, min_future=min_lead_ms / 1000.0)

        if t_hit_s is None:
            return KalmanInterceptResult(valid=False, reason="no future z-plane crossing")

        if t_hit_s * 1000.0 > max_horizon_ms:
            return KalmanInterceptResult(valid=False, reason="prediction too far in future")

        x_hit = x0 + vx * t_hit_s
        y_hit = y0 + vy * t_hit_s
        xyz_hit = np.array([x_hit, y_hit, z_catch], dtype=np.float64)

        pos_cov_trace = float(np.trace(self.P[0:3, 0:3]))
        vel_cov_trace = float(np.trace(self.P[3:6, 3:6]))
        conf = float(np.exp(-6.0 * pos_cov_trace - 1.5 * vel_cov_trace))

        return KalmanInterceptResult(
            valid=True,
            t_hit_ms=float(self.last_t_ms + 1000.0 * t_hit_s),
            xyz_hit=xyz_hit,
            xy_hit=xyz_hit[:2],
            confidence=conf,
            reason="ok",
        )

    def predict_future_points(self, n_points: int, step_ms: float) -> np.ndarray:
        if not self.initialized:
            return np.empty((0, 3), dtype=np.float64)

        pts = []
        for i in range(n_points):
            dt_s = (i * step_ms) / 1000.0
            st = self.predict_state_at_dt(dt_s)
            pts.append(st[:3])
        return np.asarray(pts, dtype=np.float64)

    def measured_points(self, n_points: int) -> np.ndarray:
        if len(self.history_xyz) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(self.history_xyz[-n_points:], dtype=np.float64)