"""
APPROACH 4: RANSAC-Robust Trajectory Fitting
============================================
RANK: #4

Core idea:
    Camera detections occasionally produce outliers – reflections, partial
    occlusions, or misidentified objects.  Polynomial/physics fitting is highly
    sensitive to even a single bad point.  RANSAC (Random Sample Consensus)
    fits the ballistic model on random minimal subsets, scores each fit by how
    many *other* observations agree with it (inliers), and keeps the best fit.

    Concretely:
        • Minimal sample for a ballistic fit = 4 points (x0, vx, vy, z0, vz
          with g known).
        • Each iteration: pick 4 random observations, fit physics model,
          count inliers (residual < threshold), keep if best so far.
        • Final fit: refit on all inliers of best consensus set.
        • Catch-plane solving: same quadratic formula as Approach 1.

Why it is #4:
    • Superior to Approaches 1/2 when there are outlier measurements.
    • Slower per frame (O(k × n) fit evaluations) – mitigated by running
      only when new points arrive or on a sub-frame thread.
    • Overkill if your camera pipeline already filters bad detections.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


G_MMS2 = 9.81 * 1e-6   # m/ms²


class RansacBallFitter:
    """
    RANSAC-based ballistic trajectory estimator.

    State estimated: (x0, vx, y0, vy, z0, vz) at time t0.
    """

    def __init__(
        self,
        catch_z:          float = 0.10,
        window_size:      int   = 30,
        min_points:       int   = 6,       # need enough for RANSAC to work
        n_iter:           int   = 50,      # RANSAC iterations
        inlier_threshold: float = 0.015,   # [m] position residual threshold
        min_inlier_ratio: float = 0.5,
        min_sample:       int   = 4,
    ) -> None:
        self.catch_z          = catch_z
        self.window_size      = window_size
        self.min_points       = min_points
        self.n_iter           = n_iter
        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_sample       = min_sample

        self._t:   np.ndarray = np.empty(0)
        self._pos: np.ndarray = np.empty((0, 3))
        self._t0:  float      = 0.0
        self._params:  Optional[np.ndarray] = None   # (x0, vx, y0, vy, z0, vz)
        self._n_inliers: int  = 0

    def reset(self) -> None:
        self._t       = np.empty(0)
        self._pos     = np.empty((0, 3))
        self._params  = None
        self._n_inliers = 0

    def update(self, timestamp_ms: float, pos_m: np.ndarray) -> None:
        pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3)

        if self._t.size == 0:
            self._t0 = timestamp_ms
        self._t   = np.append(self._t, timestamp_ms)
        self._pos = (np.vstack([self._pos, pos_m])
                     if self._pos.size else pos_m.reshape(1, 3))

        if self._t.size > self.window_size:
            self._t   = self._t[-self.window_size:]
            self._t0  = self._t[0]
            self._pos = self._pos[-self.window_size:]

        if self._t.size >= self.min_points:
            self._run_ransac()

    def predict_interception(self, now_ms: float) -> Optional[np.ndarray]:
        if self._params is None:
            return None
        t_hit = self._solve_catch_plane(now_ms)
        if t_hit is None:
            return None
        return self._eval(t_hit)

    def predict_pos(self, timestamp_ms: float) -> np.ndarray:
        if self._params is None:
            return self._pos[-1] if self._pos.size else np.zeros(3)
        return self._eval(timestamp_ms)

    # ── private ────────────────────────────────────────────────────────────

    def _run_ransac(self) -> None:
        ts  = self._t - self._t0
        pos = self._pos
        n   = len(ts)
        rng = np.random.default_rng(seed=42)

        best_params   = None
        best_inliers  = np.zeros(n, dtype=bool)
        best_n_inliers = 0

        for _ in range(self.n_iter):
            # Sample minimal set
            idx = rng.choice(n, self.min_sample, replace=False)
            try:
                params = self._fit_subset(ts[idx], pos[idx])
            except np.linalg.LinAlgError:
                continue

            # Score
            residuals  = self._residuals(ts, pos, params)
            inliers    = residuals < self.inlier_threshold
            n_inliers  = int(inliers.sum())

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers   = inliers
                best_params    = params

        # Refit on inliers
        if best_params is not None and best_n_inliers >= self.min_sample:
            ratio = best_n_inliers / n
            if ratio >= self.min_inlier_ratio:
                try:
                    self._params    = self._fit_subset(
                        ts[best_inliers], pos[best_inliers]
                    )
                    self._n_inliers = best_n_inliers
                    print(f"[RANSAC] inliers={best_n_inliers}/{n} ({100*ratio:.0f}%)")
                    return
                except np.linalg.LinAlgError:
                    pass

        # Fallback: use all points
        try:
            self._params = self._fit_subset(ts, pos)
            self._n_inliers = n
            print("[RANSAC] fallback: used all points")
        except np.linalg.LinAlgError:
            pass

    @staticmethod
    def _fit_subset(ts: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Fit ballistic parameters to a subset of (ts, pos).
        Returns (x0, vx, y0, vy, z0, vz) via linear least-squares.

        x(t) = x0 + vx*t                 (linear)
        y(t) = y0 + vy*t                 (linear)
        z(t) = z0 + vz*t - 0.5*G*t^2    (gravity subtracted as linear)
        """
        if len(ts) < 2:
            raise np.linalg.LinAlgError("Too few points")

        A = np.column_stack([np.ones_like(ts), ts])   # [1, t]

        # X
        cx, *_ = np.linalg.lstsq(A, pos[:, 0], rcond=None)
        # Y
        cy, *_ = np.linalg.lstsq(A, pos[:, 1], rcond=None)
        # Z: subtract gravity first
        z_lin  = pos[:, 2] + 0.5 * G_MMS2 * ts ** 2
        cz, *_ = np.linalg.lstsq(A, z_lin, rcond=None)

        return np.array([cx[1], cx[0],   # vx, x0  (polyfit order: slope, intercept)
                         cy[1], cy[0],
                         cz[1], cz[0],])
        # params layout: [vx, x0, vy, y0, vz, z0]

    @staticmethod
    def _residuals(ts: np.ndarray, pos: np.ndarray, params: np.ndarray) -> np.ndarray:
        vx, x0, vy, y0, vz, z0 = params
        px = x0 + vx * ts
        py = y0 + vy * ts
        pz = z0 + vz * ts - 0.5 * G_MMS2 * ts ** 2
        diff = pos - np.column_stack([px, py, pz])
        return np.linalg.norm(diff, axis=1)

    def _eval(self, timestamp_ms: float) -> np.ndarray:
        vx, x0, vy, y0, vz, z0 = self._params
        t = timestamp_ms - self._t0
        return np.array([
            x0 + vx * t,
            y0 + vy * t,
            z0 + vz * t - 0.5 * G_MMS2 * t ** 2,
        ])

    def _solve_catch_plane(self, now_ms: float) -> Optional[float]:
        _, _, _, _, vz, z0 = self._params
        a = -0.5 * G_MMS2
        b = vz
        c = z0 - self.catch_z

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        now_shift = now_ms - self._t0
        s1 = (-b + sqrt_disc) / (2 * a)
        s2 = (-b - sqrt_disc) / (2 * a)
        future = [s for s in (s1, s2) if s > now_shift]
        if not future:
            return None
        return float(self._t0 + min(future))


# ── Integration shim ──────────────────────────────────────────────────────────

def ballXYZ_to_phi_cmd_ransac(
    arm,
    XYZ:        np.ndarray,
    ball_found: bool,
    timestamp:  float,
    catch_z:    float = 0.10,
) -> np.ndarray:
    """
    Drop-in replacement for arm.ballXYZ_to_phi_cmd using RANSAC-robust fitting.

    Usage in Arm.__init__:
        self.ransac_fitter = RansacBallFitter(catch_z=0.10)
    """
    if not hasattr(arm, 'ransac_fitter'):
        arm.ransac_fitter = RansacBallFitter(catch_z=catch_z)

    if not ball_found:
        arm.missed_frames += 1
        if arm.missed_frames >= arm.missed_frames_max:
            arm.ransac_fitter.reset()
        return arm.prev_phi_cmd

    arm.missed_frames = 0
    xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(xyz)):
        return arm.prev_phi_cmd

    arm.ransac_fitter.update(timestamp, xyz)
    arm.traj.update_trajectory(timestamp, XYZ, arm._pos_q_max)

    intercept = arm.ransac_fitter.predict_interception(timestamp)
    if intercept is None:
        arm.interception_point_ROBOT = None
        return arm.prev_phi_cmd

    print(f"[ransac] intercept={intercept}, inliers={arm.ransac_fitter._n_inliers}")
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
