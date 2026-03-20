"""
APPROACH 1: Physics-First Ballistic Interception
=================================================
RANK: #1 (Best overall)

Core idea:
    Use true projectile-motion physics (constant g = -9.81 m/s²) instead of
    polynomial regression.  From the first three observations we estimate the
    launch velocity via a least-squares fit that enforces  z(t) = z0 + vz*t - 0.5*g*t².
    XY are also fit as linear functions of time (constant horizontal velocity).
    The closed-form root of the z-equation directly gives t_hit for any catch
    plane height, which we then substitute into x(t) and y(t) for the XY
    interception point.

Why it is the best:
    • Physically principled – gravity is exact, not a soft constraint.
    • Extremely stable: only 3 free parameters (x0, vx, vy, z0, vz).
    • Fast catch-plane solving: quadratic formula → O(1).
    • Naturally rejects non-physical fits (e.g. upward-only arcs once the
      physics is anchored).
    • Low latency: updates every frame once ≥3 points are available.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


G_MS2 = 9.81          # [m/s²]  gravitational acceleration (positive down)
G_MMS2 = G_MS2 * 1e-6  # [m/ms²] for when time is in milliseconds


@dataclass
class BallisticInterceptor:
    """
    Maintains a running estimate of ball trajectory under constant gravity
    and finds where it intersects a horizontal catch plane at z = catch_z.

    Units:
        positions  – metres
        timestamps – milliseconds
    """
    catch_z: float = 0.10          # [m] height of the catch plane above robot base
    window_size: int = 70          # maximum number of observations to keep
    min_points: int = 10            # minimum observations before prediction is valid

    # ── internal state ─────────────────────────────────────────────────────
    _t:   np.ndarray = field(default_factory=lambda: np.empty(0))
    _pos: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    _t0:  float      = field(default=0.0)   # time origin for numerical stability [ms]

    # latest fit parameters
    _vx: float = 0.0   # [m/ms]
    _vy: float = 0.0
    _vz: float = 0.0   # [m/ms]  (positive = upward at launch)
    _x0: float = 0.0   # [m]
    _y0: float = 0.0
    _z0: float = 0.0

    _valid: bool = False   # True once a stable fit exists

    # ── public API ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Discard all observations and reset the estimator."""
        self._t   = np.empty(0)
        self._pos = np.empty((0, 3))
        self._t0  = 0.0
        self._valid = False

    def update(self, timestamp_ms: float, pos_m: np.ndarray) -> None:
        """
        Add a new observation and re-fit the trajectory.

        Args:
            timestamp_ms: frame timestamp in milliseconds.
            pos_m:        ball position [x, y, z] in metres (robot base frame).
        """
        pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3)

        # Append
        if self._t.size == 0:
            self._t0 = timestamp_ms
        self._t   = np.append(self._t, timestamp_ms)
        self._pos = np.vstack([self._pos, pos_m]) if self._pos.size else pos_m.reshape(1, 3)

        # Window
        if self._t.size > self.window_size:
            self._t   = self._t[-self.window_size:]
            self._pos = self._pos[-self.window_size:]
            self._t0  = self._t[0]

        # Fit
        if self._t.size >= self.min_points:
            self._fit()
        else:
            self._valid = False

    def predict_interception(self, now_ms: float) -> Optional[np.ndarray]:
        """
        Compute the [x, y, z] catch point on the plane z = self.catch_z.

        Returns None if:
            • fewer than min_points observations have been seen, OR
            • the ball never reaches (or has already passed) the catch plane, OR
            • only imaginary roots exist.

        Args:
            now_ms: current time in milliseconds (used to filter past roots).

        Returns:
            np.ndarray of shape (3,) with the interception point, or None.
        """
        if not self._valid:
            return None

        t_hit_ms = self._solve_catch_plane(now_ms)
        if t_hit_ms is None:
            return None

        dt = t_hit_ms - self._t0
        x  = self._x0 + self._vx * dt
        y  = self._y0 + self._vy * dt
        z  = self._catch_z_check(dt)   # sanity – should equal catch_z
        return np.array([x, y, z], dtype=np.float64)

    def predict_pos(self, timestamp_ms: float) -> np.ndarray:
        """Evaluate the fitted trajectory at an arbitrary time (for visualisation)."""
        if not self._valid:
            return self._pos[-1] if self._pos.size else np.zeros(3)
        dt = timestamp_ms - self._t0
        x = self._x0 + self._vx * dt
        y = self._y0 + self._vy * dt
        z = self._z0 + self._vz * dt - 0.5 * G_MMS2 * dt ** 2
        return np.array([x, y, z], dtype=np.float64)

    # ── private helpers ────────────────────────────────────────────────────

    def _fit(self) -> None:
        """
        Least-squares fit of:
            x(t) = x0 + vx * (t - t0)     [linear]
            y(t) = y0 + vy * (t - t0)     [linear]
            z(t) = z0 + vz * (t - t0) - 0.5 * G * (t - t0)^2
                       [quadratic with FIXED leading coefficient = -0.5*G]

        This removes one degree of freedom from z (gravity is known exactly),
        making the fit far more stable than a free quadratic.
        """
        ts = self._t - self._t0   # shifted time [ms]

        # ── X and Y: simple linear fit ──────────────────────────────────
        cx = np.polyfit(ts, self._pos[:, 0], 1)   # [vx, x0]
        cy = np.polyfit(ts, self._pos[:, 1], 1)   # [vy, y0]
        self._vx, self._x0 = float(cx[0]), float(cx[1])
        self._vy, self._y0 = float(cy[0]), float(cy[1])

        # ── Z: linear fit of residual after subtracting gravity arc ─────
        # z_meas = z0 + vz*t - 0.5*G*t^2
        # z_meas + 0.5*G*t^2 = z0 + vz*t   <- linear in (z0, vz)
        z_resid = self._pos[:, 2] + 0.5 * G_MMS2 * ts ** 2
        cz = np.polyfit(ts, z_resid, 1)            # [vz, z0]
        self._vz, self._z0 = float(cz[0]), float(cz[1])

        self._valid = True

    def _solve_catch_plane(self, now_ms: float) -> Optional[float]:
        """
        Solve  z(t) = catch_z  for absolute time t.
            z0 + vz*(t-t0) - 0.5*G*(t-t0)^2 = catch_z
        Let  s = t - t0  (time shift):
            -0.5*G*s^2 + vz*s + (z0 - catch_z) = 0

        Returns the smallest future absolute time when ball is at catch_z,
        or None if no valid future root exists or root is unreasonably far away.
        """
        MAX_LOOKAHEAD_MS = 5000.0   # [ms] reject roots more than 5 s in the future

        a = -0.5 * G_MMS2
        b = self._vz
        c = self._z0 - self.catch_z

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None   # ball never reaches catch plane

        sqrt_disc = np.sqrt(disc)
        s1 = (-b + sqrt_disc) / (2 * a)
        s2 = (-b - sqrt_disc) / (2 * a)

        now_shift = now_ms - self._t0
        future = [s for s in (s1, s2) if now_shift < s <= now_shift + MAX_LOOKAHEAD_MS]

        if not future:
            return None

        s_hit = min(future)                # earliest future crossing
        return float(self._t0 + s_hit)

    def _catch_z_check(self, dt: float) -> float:
        return self._z0 + self._vz * dt - 0.5 * G_MMS2 * dt ** 2


# ── Integration shim ──────────────────────────────────────────────────────────
# Drop this method into Arm class (or call from ballXYZ_to_phi_cmd).

def ballXYZ_to_phi_cmd_ballistic(
    arm,                          # your Arm instance
    XYZ: np.ndarray,
    ball_found: bool,
    timestamp: float,             # milliseconds
    catch_z: float = 0.10,        # metres
) -> np.ndarray:
    """
    Replacement for arm.ballXYZ_to_phi_cmd that uses physics-first ballistic
    interception.

    Usage in Arm.__init__:
        self.ballistic = BallisticInterceptor(catch_z=0.10)

    Usage (call instead of arm.ballXYZ_to_phi_cmd):
        phi_cmd = ballXYZ_to_phi_cmd_ballistic(arm, XYZ, ball_found, timestamp)
    """
    if not hasattr(arm, 'ballistic'):
        arm.ballistic = BallisticInterceptor(catch_z=catch_z)

    if not ball_found:
        arm.missed_frames += 1
        if arm.missed_frames >= arm.missed_frames_max:
            arm.ballistic.reset()
            arm.traj = type(arm.traj)()   # fresh Trajectory too
        return arm.prev_phi_cmd

    arm.missed_frames = 0
    xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(xyz)):
        return arm.prev_phi_cmd

    arm.ballistic.update(timestamp, xyz)
    arm.traj.update_trajectory(timestamp, XYZ, arm._pos_q_max)  # keep traj for visualisation

    intercept = arm.ballistic.predict_interception(timestamp)
    if intercept is None:
        print("[ballistic] no valid interception point yet")
        arm.interception_point_ROBOT = None
        arm.interception_time = None
        return arm.prev_phi_cmd

    print(f"[ballistic] intercept XYZ = {intercept}")
    arm.interception_point_ROBOT = intercept

    ik_all_solns, ik_soln = arm.qarm_inverse_kinematics(intercept, 0, arm.phi)
    phi_seed = np.asarray(arm.phi, dtype=np.float64)
    all_solns = np.asarray(ik_all_solns, dtype=np.float64)

    best_phi = ik_soln.copy()
    best_dist = np.linalg.norm(best_phi - phi_seed)
    for col in range(all_solns.shape[1]):
        cand = all_solns[:, col]
        if np.all(np.isfinite(cand)):
            d = np.linalg.norm(cand - phi_seed)
            if d < best_dist:
                best_dist = d
                best_phi = cand

    arm.phi_cmd = best_phi.copy()
    arm.pos_cmd = intercept.copy()
    arm.prev_phi_cmd = best_phi.copy()
    arm.move(phi_Cmd=best_phi)
    return best_phi
