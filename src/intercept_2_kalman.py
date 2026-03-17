"""
APPROACH 2: Kalman Filter Trajectory Estimator
===============================================
RANK: #2

Core idea:
    A 6-state Kalman filter tracks [x, y, z, vx, vy, vz].  The process model
    encodes constant horizontal velocity and constant vertical deceleration
    from gravity:

        x(k+1)  = x(k)  + vx(k)*dt
        y(k+1)  = y(k)  + vy(k)*dt
        z(k+1)  = z(k)  + vz(k)*dt - 0.5*g*dt^2    ← deterministic gravity term
        vx(k+1) = vx(k)
        vy(k+1) = vy(k)
        vz(k+1) = vz(k) - g*dt                      ← gravity updates vz

    The camera measurement gives [x, y, z] directly, so H = [I₃ | 0₃].

Why it is #2:
    • Optimal under Gaussian noise – handles measurement noise gracefully.
    • Self-corrects velocity estimates over time; pure regression can't do this
      when measurements are noisy or arrive at irregular intervals.
    • Covariance gives a principled confidence measure (can gate bad frames).
    • Slightly more complex to tune (Q, R matrices) than approach #1.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


G_MS2  = 9.81           # m/s²  gravity
G_MMS2 = G_MS2 * 1e-6   # m/ms²


class KalmanBallTracker:
    """
    6-state Kalman filter for a ball under gravity.

    State  x_k = [px, py, pz, vx, vy, vz]^T   (SI + ms time)
    Obs    z_k = [px, py, pz]^T
    """

    def __init__(
        self,
        catch_z:       float = 0.10,    # [m] catch-plane height
        sigma_pos:     float = 0.005,   # [m]  measurement std (camera noise)
        sigma_process: float = 0.002,   # [m/ms²]  process acceleration noise
        min_obs:       int   = 3,
    ) -> None:
        self.catch_z       = catch_z
        self.sigma_pos     = sigma_pos
        self.sigma_process = sigma_process
        self.min_obs       = min_obs

        self._initialized  = False
        self._n_obs        = 0
        self._t_prev       = 0.0

        # State and covariance
        self.x  = np.zeros(6)
        self.P  = np.eye(6) * 1e6    # large initial uncertainty

        # Measurement matrix H: observe positions only
        self.H  = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.R  = np.eye(3) * sigma_pos ** 2

    # ── public API ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._initialized = False
        self._n_obs       = 0
        self.x  = np.zeros(6)
        self.P  = np.eye(6) * 1e6

    def update(self, timestamp_ms: float, pos_m: np.ndarray) -> None:
        """
        Predict to *timestamp_ms* then correct with *pos_m*.

        On the very first call we warm-start position from the measurement
        and set velocity to zero (with huge variance so it quickly corrects).
        """
        pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3)

        if not self._initialized:
            self.x[:3] = pos_m
            self.x[3:] = 0.0
            self.P = np.diag([sigma**2 for sigma in
                              [self.sigma_pos]*3 + [1.0]*3])
            self._t_prev      = timestamp_ms
            self._initialized = True
            self._n_obs       = 1
            return

        dt = (timestamp_ms - self._t_prev) * 1.0   # [ms]
        self._t_prev = timestamp_ms

        # ── predict ────────────────────────────────────────────────────
        x_pred = self._state_transition(self.x, dt)
        F      = self._F(dt)
        Q      = self._process_noise(dt)
        P_pred = F @ self.P @ F.T + Q

        # ── update ─────────────────────────────────────────────────────
        y  = pos_m - self.H @ x_pred          # innovation
        S  = self.H @ P_pred @ self.H.T + self.R
        K  = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred
        self._n_obs += 1

    def predict_interception(self, now_ms: float) -> Optional[np.ndarray]:
        """
        Predict where the ball will be when z = catch_z.

        Returns None if the filter hasn't been initialised or no valid future
        root exists.
        """
        if not self._initialized or self._n_obs < self.min_obs:
            return None

        t_hit = self._solve_catch_plane(now_ms)
        if t_hit is None:
            return None

        future_state = self._state_at(t_hit)
        return future_state[:3].copy()

    def predict_pos(self, timestamp_ms: float) -> np.ndarray:
        """Position prediction at an arbitrary future time (for visualisation)."""
        if not self._initialized:
            return np.zeros(3)
        return self._state_at(timestamp_ms)[:3]

    # ── private ────────────────────────────────────────────────────────────

    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Nonlinear state transition (gravity in vz and pz)."""
        px, py, pz, vx, vy, vz = x
        return np.array([
            px + vx * dt,
            py + vy * dt,
            pz + vz * dt - 0.5 * G_MMS2 * dt ** 2,
            vx,
            vy,
            vz - G_MMS2 * dt,
        ])

    def _state_at(self, future_ms: float) -> np.ndarray:
        """Propagate current state to *future_ms* without a measurement."""
        dt = future_ms - self._t_prev
        return self._state_transition(self.x, dt)

    def _F(self, dt: float) -> np.ndarray:
        """Linearised state-transition Jacobian (linear-gravity model)."""
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # vz row: vz(k+1) = vz(k) - g*dt   → d/d(vz) = 1 (no state coupling via gravity)
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Discrete-time process noise for a constant-velocity model.
        Treat each axis as a position-velocity pair driven by white acceleration noise.
        """
        q  = self.sigma_process ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        block = q * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q = np.zeros((6, 6))
        for i, j in [(0, 3), (1, 4), (2, 5)]:
            Q[np.ix_([i, j], [i, j])] += block
        return Q

    def _solve_catch_plane(self, now_ms: float) -> Optional[float]:
        """
        Solve z(t) = catch_z using current state.
        Uses second-order Taylor expansion from current state.

        z(t_prev + s)  = pz + vz*s - 0.5*G*s^2  =  catch_z
        =>  -0.5*G*s² + vz*s + (pz - catch_z) = 0
        """
        MAX_LOOKAHEAD_MS = 5000.0   # reject roots more than 5 s ahead

        pz = self.x[2]
        vz = self.x[5]

        a = -0.5 * G_MMS2
        b = vz
        c = pz - self.catch_z

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        now_shift = now_ms - self._t_prev   # how far ahead is "now" from last update
        s1 = (-b + sqrt_disc) / (2 * a)
        s2 = (-b - sqrt_disc) / (2 * a)
        future = [s for s in (s1, s2) if now_shift < s <= now_shift + MAX_LOOKAHEAD_MS]
        if not future:
            return None

        s_hit = min(future)
        return float(self._t_prev + s_hit)


# ── Integration shim ──────────────────────────────────────────────────────────

def ballXYZ_to_phi_cmd_kalman(
    arm,
    XYZ:         np.ndarray,
    ball_found:  bool,
    timestamp:   float,
    catch_z:     float = 0.10,
    sigma_pos:   float = 0.005,
    sigma_proc:  float = 0.002,
) -> np.ndarray:
    """
    Drop-in replacement for arm.ballXYZ_to_phi_cmd using a Kalman filter.

    Usage in Arm.__init__:
        self.kf_tracker = KalmanBallTracker(catch_z=0.10)
    """
    if not hasattr(arm, 'kf_tracker'):
        arm.kf_tracker = KalmanBallTracker(
            catch_z=catch_z,
            sigma_pos=sigma_pos,
            sigma_process=sigma_proc,
        )

    if not ball_found:
        arm.missed_frames += 1
        if arm.missed_frames >= arm.missed_frames_max:
            arm.kf_tracker.reset()
        return arm.prev_phi_cmd

    arm.missed_frames = 0
    xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(xyz)):
        return arm.prev_phi_cmd

    arm.kf_tracker.update(timestamp, xyz)
    arm.traj.update_trajectory(timestamp, XYZ, arm._pos_q_max)

    intercept = arm.kf_tracker.predict_interception(timestamp)
    if intercept is None:
        print("[kalman] no valid interception point yet")
        arm.interception_point_ROBOT = None
        return arm.prev_phi_cmd

    print(f"[kalman] intercept XYZ = {intercept}")
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
                best_phi  = cand

    arm.phi_cmd      = best_phi.copy()
    arm.pos_cmd      = intercept.copy()
    arm.prev_phi_cmd = best_phi.copy()
    arm.move(phi_Cmd=best_phi)
    return best_phi
