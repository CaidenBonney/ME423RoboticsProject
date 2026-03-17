

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from KalmanFilter import LinearKalmanFilter


@dataclass
class PlaneIntersection:
    time_to_hit: float
    point: np.ndarray           # shape (3,)
    state_at_hit: np.ndarray    # shape (6,)


class TrajEstimator:
    """
    High-level trajectory estimator for a flying ball.

    Owns a LinearKalmanFilter instance and provides methods for:
      - ingesting new measurements
      - querying current filtered state
      - predicting trajectory forward
      - intersecting trajectory with a fixed z-plane
      - checking basic catch feasibility
    """

    def __init__(
        self,
        kf: LinearKalmanFilter,
        workspace_xyz_min: Optional[np.ndarray] = None,
        workspace_xyz_max: Optional[np.ndarray] = None,
        min_prediction_time: float = 0.0,
        max_prediction_time: float = 2.0,
    ):
        """
        Args:
            kf: instance of LinearKalmanFilter
            gravity: positive scalar, gravitational acceleration magnitude
            workspace_xyz_min: optional [xmin, ymin, zmin]
            workspace_xyz_max: optional [xmax, ymax, zmax]
            min_prediction_time: reject intersections sooner than this
            max_prediction_time: reject intersections later than this
        """
        self.kf = kf
        self.g = self.kf.g

        self.workspace_xyz_min = (
            np.asarray(workspace_xyz_min, dtype=float).reshape(3)
            if workspace_xyz_min is not None else None
        )
        self.workspace_xyz_max = (
            np.asarray(workspace_xyz_max, dtype=float).reshape(3)
            if workspace_xyz_max is not None else None
        )

        self.min_prediction_time = float(min_prediction_time)
        self.max_prediction_time = float(max_prediction_time)

        self.last_measurement = None
        self.last_timestamp = None
        self.num_updates = 0

    # -----------------------------
    # Measurement / filter interface
    # -----------------------------
    def update(self, measurement_xyz, timestamp: float) -> np.ndarray:
        """
        Update the underlying KF with a new 3D measurement.

        Args:
            measurement_xyz: iterable [x, y, z]
            timestamp: measurement time in seconds

        Returns:
            Current filtered state [x, y, z, vx, vy, vz]
        """
        measurement_xyz = np.asarray(measurement_xyz, dtype=float).reshape(3)
        state = np.asarray(self.kf.step(measurement_xyz, timestamp), dtype=float).reshape(6)

        self.last_measurement = measurement_xyz
        self.last_timestamp = float(timestamp)
        self.num_updates += 1
        return state

    def initialize(self, measurement_xyz, timestamp: float, velocity_xyz=None) -> None:
        """
        Explicitly initialize the KF.
        """
        measurement_xyz = np.asarray(measurement_xyz, dtype=float).reshape(3)
        if velocity_xyz is not None:
            velocity_xyz = np.asarray(velocity_xyz, dtype=float).reshape(3)

        self.kf.initialize(measurement_xyz, timestamp, velocity=velocity_xyz)
        self.last_measurement = measurement_xyz
        self.last_timestamp = float(timestamp)
        self.num_updates = 0

    def is_initialized(self) -> bool:
        return getattr(self.kf, "initialized", False)

    def get_state(self) -> Optional[np.ndarray]:
        """
        Returns current filtered state [x, y, z, vx, vy, vz], or None if unavailable.
        """
        if not self.is_initialized():
            return None
        return np.asarray(self.kf.current_state(), dtype=float).reshape(6)

    def get_position(self) -> Optional[np.ndarray]:
        state = self.get_state()
        return None if state is None else state[:3].copy()

    def get_velocity(self) -> Optional[np.ndarray]:
        state = self.get_state()
        return None if state is None else state[3:].copy()

    # -----------------------------
    # Forward prediction
    # -----------------------------
    def predict_state_dt(self, dt: float, state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Predict state forward by dt seconds using ballistic dynamics.

        Args:
            dt: time forward in seconds
            state: optional starting state; defaults to current filtered state

        Returns:
            Predicted state [x, y, z, vx, vy, vz]
        """
        if dt < 0.0:
            return None

        if state is None:
            state = self.get_state()
        if state is None:
            return None

        x0, y0, z0, vx, vy, vz = map(float, state)

        x = x0 + vx * dt
        y = y0 + vy * dt
        z = z0 + vz * dt - 0.5 * self.g * dt * dt

        vz_new = vz - self.g * dt

        return np.array([x, y, z, vx, vy, vz_new], dtype=float)

    def predict_position_dt(self, dt: float, state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        pred = self.predict_state_dt(dt, state=state)
        return None if pred is None else pred[:3]

    # -----------------------------
    # Plane intersection
    # -----------------------------
    def intersect_z_plane(
        self,
        z_plane: float,
        state: Optional[np.ndarray] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> Optional[PlaneIntersection]:
        """
        Intersect predicted ballistic trajectory with plane z = z_plane.

        Args:
            z_plane: target z height
            state: optional state [x,y,z,vx,vy,vz], defaults to current filtered state
            t_min: optional lower bound on valid future hit time
            t_max: optional upper bound on valid future hit time

        Returns:
            PlaneIntersection or None if no valid future intersection exists
        """
        if state is None:
            state = self.get_state()
        if state is None:
            return None

        if t_min is None:
            t_min = self.min_prediction_time
        if t_max is None:
            t_max = self.max_prediction_time

        x0, y0, z0, vx, vy, vz = map(float, state)

        # Solve z_plane = z0 + vz*t - 0.5*g*t^2
        a = -0.5 * self.g
        b = vz
        c = z0 - float(z_plane)

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t2 = (-b - sqrt_disc) / (2.0 * a)

        candidates = [t for t in (t1, t2) if t > 0.0]
        if t_min is not None:
            candidates = [t for t in candidates if t >= t_min]
        if t_max is not None:
            candidates = [t for t in candidates if t <= t_max]

        if not candidates:
            return None

        t_hit = min(candidates)
        state_hit = self.predict_state_dt(t_hit, state=state)
        point_hit = state_hit[:3]

        return PlaneIntersection(
            time_to_hit=t_hit,
            point=point_hit,
            state_at_hit=state_hit,
        )

    # -----------------------------
    # Feasibility / workspace checks
    # -----------------------------
    def is_point_in_workspace(self, point_xyz) -> bool:
        point_xyz = np.asarray(point_xyz, dtype=float).reshape(3)

        if self.workspace_xyz_min is not None and np.any(point_xyz < self.workspace_xyz_min):
            return False
        if self.workspace_xyz_max is not None and np.any(point_xyz > self.workspace_xyz_max):
            return False
        return True

    def is_intersection_feasible(
        self,
        intersection: Optional[PlaneIntersection],
        robot_min_time: Optional[float] = None,
    ) -> bool:
        """
        Basic feasibility check for a predicted catch point.
        """
        if intersection is None:
            return False

        if robot_min_time is not None and intersection.time_to_hit < robot_min_time:
            return False

        if not self.is_point_in_workspace(intersection.point):
            return False

        return True

    def estimate_catch(
        self,
        z_plane: float,
        robot_min_time: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience method:
          - uses current filtered state
          - intersects with z-plane
          - performs simple feasibility checks

        Returns:
            dict with catch estimate or None
        """
        intersection = self.intersect_z_plane(z_plane)
        if not self.is_intersection_feasible(intersection, robot_min_time=robot_min_time):
            return None

        return {
            "time_to_hit": intersection.time_to_hit,
            "catch_point": intersection.point.copy(),
            "state_at_hit": intersection.state_at_hit.copy(),
            "current_state": self.get_state().copy(),
        }

    # -----------------------------
    # Debug / status helpers
    # -----------------------------
    def summary(self) -> Dict[str, Any]:
        state = self.get_state()
        return {
            "initialized": self.is_initialized(),
            "num_updates": self.num_updates,
            "last_timestamp": self.last_timestamp,
            "last_measurement": None if self.last_measurement is None else self.last_measurement.copy(),
            "state": None if state is None else state.copy(),
        }


# Your existing filter class
kf = LinearKalmanFilter(
    g=9.81,
    accel_noise_std=0.5,
    meas_noise_std=0.005,
)

traj = TrajEstimator(
    kf=kf,
    workspace_xyz_min=np.array([0.2, -0.5, 0.4]),
    workspace_xyz_max=np.array([1.2,  0.5, 1.2]),
    min_prediction_time=0.05,
    max_prediction_time=1.5,
)

# In your camera callback / loop:
measurement_xyz = [0.81, 0.10, 1.45]
timestamp = 12.384

state = traj.update(measurement_xyz, timestamp)

catch = traj.estimate_catch(z_plane=0.45, robot_min_time=0.18)
if catch is not None:
    print("time_to_hit:", catch["time_to_hit"])
    print("catch_point:", catch["catch_point"])
    print("state_at_hit:", catch["state_at_hit"])