"""
APPROACH 5: EWM Velocity Estimator + Plane Intersection
========================================================
RANK: #5

Core idea:
    The simplest viable approach.  No fitting, no filter – just maintain an
    exponentially weighted moving average (EWMA) of ball velocity and use the
    current position + smoothed velocity to extrapolate when z = catch_z.

    Velocity at frame k:
        v_raw(k) = (pos(k) - pos(k-1)) / dt(k)
        v_ewm(k) = α * v_raw(k) + (1 - α) * v_ewm(k-1)

    Then z(t) = z_now + vz_ewm * t - 0.5*g*t^2 = catch_z  is solved for t,
    and  x,y  are evaluated linearly at that t.

    No covariance tracking, no buffer of all historical points – O(1) memory
    and O(1) CPU per frame.

Why it is #5:
    • Extremely fast and simple – zero tuning beyond α.
    • Accurate for slow, smooth ball trajectories with low noise.
    • Velocity estimate is noisy for fast-moving balls sampled at low frame
      rates – smoothing helps but can't fully overcome this.
    • No outlier rejection.  Good as a baseline or warm-start seed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


G_MMS2 = 9.81 * 1e-6   # m/ms²


class EWMAInterceptor:
    """
    Lightweight EWMA velocity estimator + ballistic plane intersection.
    """

    def __init__(
        self,
        catch_z: float = 0.10,
        alpha:   float = 0.4,     # EWMA weight for new velocity sample [0, 1]
                                   # larger = faster response, noisier
        min_obs: int   = 3,
    ) -> None:
        self.catch_z = catch_z
        self.alpha   = alpha
        self.min_obs = min_obs

        self._pos_prev:  Optional[np.ndarray] = None
        self._t_prev:    Optional[float]       = None
        self._vel_ewma:  np.ndarray            = np.zeros(3)
        self._pos_now:   np.ndarray            = np.zeros(3)
        self._n_obs:     int                   = 0

    def reset(self) -> None:
        self._pos_prev = None
        self._t_prev   = None
        self._vel_ewma = np.zeros(3)
        self._n_obs    = 0

    def update(self, timestamp_ms: float, pos_m: np.ndarray) -> None:
        pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3)
        self._pos_now = pos_m.copy()

        if self._pos_prev is not None and self._t_prev is not None:
            dt = timestamp_ms - self._t_prev
            if dt > 0:
                v_raw = (pos_m - self._pos_prev) / dt   # [m/ms]
                self._vel_ewma = (
                    self.alpha * v_raw + (1 - self.alpha) * self._vel_ewma
                )

        self._pos_prev = pos_m.copy()
        self._t_prev   = timestamp_ms
        self._n_obs   += 1

    def predict_interception(self, now_ms: float) -> Optional[np.ndarray]:
        if self._n_obs < self.min_obs:
            return None

        t_hit = self._solve_catch_plane()
        if t_hit is None:
            return None

        x = self._pos_now[0] + self._vel_ewma[0] * t_hit
        y = self._pos_now[1] + self._vel_ewma[1] * t_hit
        return np.array([x, y, self.catch_z], dtype=np.float64)

    def predict_pos(self, dt_ms: float) -> np.ndarray:
        """Position dt_ms milliseconds from the last observation."""
        vx, vy, vz = self._vel_ewma
        px, py, pz = self._pos_now
        return np.array([
            px + vx * dt_ms,
            py + vy * dt_ms,
            pz + vz * dt_ms - 0.5 * G_MMS2 * dt_ms ** 2,
        ])

    # ── private ────────────────────────────────────────────────────────────

    def _solve_catch_plane(self) -> Optional[float]:
        """
        Solve z_now + vz*t - 0.5*g*t^2 = catch_z for t > 0.
        Returns smallest positive real root within 5 s, or None.
        """
        MAX_LOOKAHEAD_MS = 5000.0   # reject roots more than 5 s ahead

        pz = self._pos_now[2]
        vz = self._vel_ewma[2]

        a = -0.5 * G_MMS2
        b = vz
        c = pz - self.catch_z

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        s1 = (-b + sqrt_disc) / (2 * a)
        s2 = (-b - sqrt_disc) / (2 * a)
        future = [s for s in (s1, s2) if 0 < s <= MAX_LOOKAHEAD_MS]
        if not future:
            return None
        return float(min(future))


# ── Integration shim ──────────────────────────────────────────────────────────

def ballXYZ_to_phi_cmd_ewma(
    arm,
    XYZ:        np.ndarray,
    ball_found: bool,
    timestamp:  float,
    catch_z:    float = 0.10,
    alpha:      float = 0.4,
) -> np.ndarray:
    """
    Drop-in replacement for arm.ballXYZ_to_phi_cmd using EWMA velocity estimation.

    Usage in Arm.__init__:
        self.ewma_interceptor = EWMAInterceptor(catch_z=0.10)
    """
    if not hasattr(arm, 'ewma_interceptor'):
        arm.ewma_interceptor = EWMAInterceptor(catch_z=catch_z, alpha=alpha)

    if not ball_found:
        arm.missed_frames += 1
        if arm.missed_frames >= arm.missed_frames_max:
            arm.ewma_interceptor.reset()
        return arm.prev_phi_cmd

    arm.missed_frames = 0
    xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(xyz)):
        return arm.prev_phi_cmd

    arm.ewma_interceptor.update(timestamp, xyz)
    arm.traj.update_trajectory(timestamp, XYZ, arm._pos_q_max)

    intercept = arm.ewma_interceptor.predict_interception(timestamp)
    if intercept is None:
        arm.interception_point_ROBOT = None
        return arm.prev_phi_cmd

    print(f"[ewma] intercept={intercept}")
    arm.interception_point_ROBOT = intercept

    ik_all_solns, ik_soln = arm.qarm_inverse_kinematics(intercept, 0, arm.phi)
    phi_seed  = np.asarray(arm.phi, dtype=np.float64)
    all_solns = np.asarray(ik_all_solns, dtype=np.float64)

    best_phi = ik_soln.copy()
    best_dist = np.linalg.norm(best_phi - phi_seed)
    for col in range(all_solns.shape[1]):
        cand = all_solns[:, col]
        if np.all(np.isfinite(cand)):
            d = np.linalg.norm(cand - phi_seed)
            if d < best_dist:
                best_dist = d
                best_phi  = cand.copy()

    arm.phi_cmd      = best_phi.copy()
    arm.pos_cmd      = intercept.copy()
    arm.prev_phi_cmd = best_phi.copy()
    arm.move(phi_Cmd=best_phi)
    return best_phi
