"""
APPROACH 3: Time-Aware Reachability Interception
=================================================
RANK: #3

Core idea:
    Physics-first ballistic prediction (same as Approach 1) PLUS an explicit
    model of the arm's travel time.  Rather than commanding the arm to the
    *first* future root, we scan the ball's trajectory for the earliest catch
    position the arm can physically reach in time.

    For each candidate catch time t_c (sampled densely along the ballistic arc):
        1.  Compute ball XY on the catch plane at t_c.
        2.  Run IK → desired joint angles φ_desired.
        3.  Estimate arm travel time  T_arm  from current φ to φ_desired using
            a per-joint max-velocity limit.
        4.  Accept t_c  iff  t_c - t_now  ≥  T_arm  (arm can arrive in time)
            and choose the *smallest* such t_c.

    The result is the earliest reachable interception point, accounting for the
    arm's kinematics.

Why it is #3:
    • Only approach that explicitly accounts for arm speed – won't command a
      physically impossible interception.
    • Extra complexity costs a few ms of CPU per frame – acceptable.
    • Requires reasonable joint-velocity limits to be calibrated.
    • Slightly more conservative than #1/#2 (may miss some fast balls).
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from intercept_1_ballistic import BallisticInterceptor   # reuse physics core


# Approximate maximum joint speeds [rad/ms] for QArm
# (tune these to match your hardware spec or measure empirically)
JOINT_VEL_MAX_RAD_MS = np.array([
    np.radians(180) / 1000,   # joint 0 (base rotation)
    np.radians(90)  / 1000,   # joint 1 (shoulder)
    np.radians(90)  / 1000,   # joint 2 (elbow)
    np.radians(120) / 1000,   # joint 3 (wrist)
], dtype=np.float64)

# Safety margin: multiply estimated travel time by this factor.
# Accounts for acceleration / deceleration ramps.
ARM_TRAVEL_SAFETY_FACTOR = 1.5

# Scan resolution: how many candidate times to evaluate along the arc
N_SCAN = 80            # samples evaluated per call
SCAN_WINDOW_MS = 3000  # how far ahead to look [ms]


class ReachabilityInterceptor:
    """
    Combines ballistic trajectory prediction with arm reachability analysis.
    """

    def __init__(
        self,
        catch_z: float = 0.10,
        window_size: int = 30,
        min_points: int = 3,
        joint_vel_max: np.ndarray = JOINT_VEL_MAX_RAD_MS,
        safety_factor: float = ARM_TRAVEL_SAFETY_FACTOR,
    ) -> None:
        self.catch_z       = catch_z
        self.safety_factor = safety_factor
        self.joint_vel_max = np.asarray(joint_vel_max, dtype=np.float64)

        self._physics = BallisticInterceptor(
            catch_z=catch_z,
            window_size=window_size,
            min_points=min_points,
        )

        # Cache of last successful IK solution for continuity
        self._last_phi: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._physics.reset()
        self._last_phi = None

    def update(self, timestamp_ms: float, pos_m: np.ndarray) -> None:
        self._physics.update(timestamp_ms, pos_m)

    def predict_reachable_intercept(
        self,
        now_ms: float,
        arm,              # Arm instance for IK + current phi
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Scan the ballistic arc forward in time and return the earliest
        interception point the arm can physically reach.

        Returns:
            (intercept_xyz, phi_cmd)  – both None if no reachable point found.
        """
        if not self._physics._valid:
            return None, None

        phi_now = np.asarray(arm.phi, dtype=np.float64)
        t_candidates = np.linspace(now_ms, now_ms + SCAN_WINDOW_MS, N_SCAN)

        for t_c in t_candidates:
            # ── evaluate ball position at t_c ──────────────────────────
            pos = self._physics.predict_pos(t_c)
            if abs(pos[2] - self.catch_z) > 0.05:   # skip if too far from z-plane
                continue

            # Clamp to exact catch-plane height
            pos[2] = self.catch_z

            # ── IK ─────────────────────────────────────────────────────
            try:
                all_solns, _ = arm.qarm_inverse_kinematics(pos, 0, phi_now)
            except Exception:
                continue

            all_solns = np.asarray(all_solns, dtype=np.float64)
            phi_best  = self._best_ik(all_solns, phi_now)
            if phi_best is None:
                continue

            # ── travel time estimate ────────────────────────────────────
            delta_phi   = np.abs(phi_best - phi_now)
            t_arm_ms    = float(np.max(delta_phi / self.joint_vel_max))
            t_arm_ms   *= self.safety_factor
            time_avail  = t_c - now_ms

            if time_avail >= t_arm_ms:
                print(
                    f"[reachability] t_c={t_c:.0f} ms, avail={time_avail:.0f} ms, "
                    f"arm_travel={t_arm_ms:.0f} ms  → REACHABLE"
                )
                self._last_phi = phi_best
                return pos.copy(), phi_best

        print("[reachability] no reachable interception found in scan window")
        return None, None

    @staticmethod
    def _best_ik(all_solns: np.ndarray, phi_ref: np.ndarray) -> Optional[np.ndarray]:
        """Pick the IK solution closest (in joint space) to phi_ref."""
        best_phi = None
        best_dist = np.inf
        for col in range(all_solns.shape[1]):
            cand = all_solns[:, col]
            if not np.all(np.isfinite(cand)):
                continue
            d = np.linalg.norm(cand - phi_ref)
            if d < best_dist:
                best_dist = d
                best_phi  = cand.copy()
        return best_phi


# ── Integration shim ──────────────────────────────────────────────────────────

def ballXYZ_to_phi_cmd_reachability(
    arm,
    XYZ:        np.ndarray,
    ball_found: bool,
    timestamp:  float,
    catch_z:    float = 0.10,
) -> np.ndarray:
    """
    Drop-in replacement for arm.ballXYZ_to_phi_cmd using reachability-aware
    interception planning.

    Usage in Arm.__init__:
        self.reach_interceptor = ReachabilityInterceptor(catch_z=0.10)
    """
    if not hasattr(arm, 'reach_interceptor'):
        arm.reach_interceptor = ReachabilityInterceptor(catch_z=catch_z)

    if not ball_found:
        arm.missed_frames += 1
        if arm.missed_frames >= arm.missed_frames_max:
            arm.reach_interceptor.reset()
        return arm.prev_phi_cmd

    arm.missed_frames = 0
    xyz = np.asarray(XYZ, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(xyz)):
        return arm.prev_phi_cmd

    arm.reach_interceptor.update(timestamp, xyz)
    arm.traj.update_trajectory(timestamp, XYZ, arm._pos_q_max)

    intercept, phi_cmd = arm.reach_interceptor.predict_reachable_intercept(
        timestamp, arm
    )

    if intercept is None or phi_cmd is None:
        arm.interception_point_ROBOT = None
        return arm.prev_phi_cmd

    print(f"[reachability] intercept={intercept}, phi={np.degrees(phi_cmd).round(1)}°")
    arm.interception_point_ROBOT = intercept

    arm.phi_cmd      = phi_cmd.copy()
    arm.pos_cmd      = intercept.copy()
    arm.prev_phi_cmd = phi_cmd.copy()
    arm.move(phi_Cmd=phi_cmd)
    return phi_cmd
