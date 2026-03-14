import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

class LinearKalmanFilter:
    """
    Linear Kalman filter for ballistic flight with variable dt.
    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    Gravity acts in -z.

    g=9.81 is about 9.81m/s^2 gravity.

    accel_noise_std=0.5 is about 0.5g acceleration noise. If the filter is too sluggish, increase it. If it is too jittery, decrease it.
    meas_noise_std=0.005 is about 5mm measurement noise.
    """ 
    def __init__(self, g=9.81, accel_noise_std=0.5, meas_noise_std=0.005):
        self.g = float(g)
        self.accel_noise_std = float(accel_noise_std)
        self.meas_noise_std = float(meas_noise_std)

        self.x = np.zeros((6, 1))     # state
        self.P = np.eye(6) * 10.0     # covariance
        self.R = np.eye(3) * (self.meas_noise_std ** 2)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=float)

        self.initialized = False
        self.last_timestamp = None
        self.prev_meas = None
        self.prev_meas_time = None

    def _make_F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=float)

    def _make_B(self, dt: float) -> np.ndarray:
        return np.array([
            [0],
            [0],
            [-0.5 * dt * dt],
            [0],
            [0],
            [-dt]
        ], dtype=float)

    def _make_Q(self, dt: float) -> np.ndarray:
        """
        Process noise from white acceleration model.
        Same acceleration-noise strength on x, y, z.
        """
        q = self.accel_noise_std ** 2

        q1 = np.array([
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2]
        ], dtype=float) * q

        Q = np.zeros((6, 6), dtype=float)

        # x / vx block
        Q[np.ix_([0, 3], [0, 3])] = q1
        # y / vy block
        Q[np.ix_([1, 4], [1, 4])] = q1
        # z / vz block
        Q[np.ix_([2, 5], [2, 5])] = q1

        return Q

    def initialize(self, position, timestamp, velocity=None):
        position = np.asarray(position, dtype=float).reshape(3)
        self.x[0:3, 0] = position

        if velocity is not None:
            self.x[3:6, 0] = np.asarray(velocity, dtype=float).reshape(3)
        else:
            self.x[3:6, 0] = 0.0

        self.last_timestamp = float(timestamp)
        self.prev_meas = position.copy()
        self.prev_meas_time = float(timestamp)
        self.initialized = True

    def _predict(self, dt: float):
        F = self._make_F(dt)
        B = self._make_B(dt)
        Q = self._make_Q(dt)

        u = np.array([[self.g]], dtype=float)

        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + Q

    def _update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(3, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def step(self, measurement, timestamp):
        """
        measurement: [x, y, z]
        timestamp: seconds, monotonic preferred

        Returns:
            Predicted state [x, y, z, vx, vy, vz] after processing this measurement.
        """
        measurement = np.asarray(measurement, dtype=float).reshape(3)
        timestamp = float(timestamp)

        if not self.initialized:
            self.initialize(measurement, timestamp)
            return self.x.flatten()

        # If we still don't have a decent velocity estimate,
        # bootstrap it from the first two measurements.
        if np.allclose(self.x[3:6, 0], 0.0) and self.prev_meas is not None:
            dt_boot = timestamp - self.prev_meas_time
            if dt_boot > 1e-6:
                v0 = (measurement - self.prev_meas) / dt_boot
                self.x[3:6, 0] = v0.reshape(3)
                self.last_timestamp = timestamp
                self.prev_meas = measurement.copy()
                self.prev_meas_time = timestamp
                return self.x.flatten()

        dt = timestamp - self.last_timestamp
        if dt <= 1e-6:
            # Ignore non-forward or duplicate timestamps
            return self.x.flatten()

        self._predict(dt)
        self._update(measurement)

        self.last_timestamp = timestamp
        self.prev_meas = measurement.copy()
        self.prev_meas_time = timestamp

        return self.x.flatten()

    def current_state(self):
        return self.x.flatten()

    def predict_to_time(self, future_timestamp):
        """
        Predict state forward without measurement update.
        """
        future_timestamp = float(future_timestamp)
        if self.last_timestamp is None:
            return None

        dt = future_timestamp - self.last_timestamp
        if dt < 0:
            return None

        F = self._make_F(dt)
        B = self._make_B(dt)
        u = np.array([[self.g]], dtype=float)

        x_pred = F @ self.x + B @ u
        return x_pred.flatten()

# def intersect_with_z_plane(state, z_plane, g=9.81):
#     """
#     state = [x, y, z, vx, vy, vz]
#     returns (t_hit, point) where point = [x_hit, y_hit, z_plane]
#     t_hit is measured forward from the current state time.
#     """
#     x0, y0, z0, vx, vy, vz = map(float, state)

#     # z_plane = z0 + vz*t - 0.5*g*t^2
#     a = -0.5 * g
#     b = vz
#     c = z0 - z_plane

#     disc = b * b - 4.0 * a * c
#     if disc < 0:
#         return None

#     sqrt_disc = np.sqrt(disc)
#     t1 = (-b + sqrt_disc) / (2.0 * a)
#     t2 = (-b - sqrt_disc) / (2.0 * a)

#     positive_times = [t for t in (t1, t2) if t > 0.0]
#     if not positive_times:
#         return None

#     t_hit = min(positive_times)

#     x_hit = x0 + vx * t_hit
#     y_hit = y0 + vy * t_hit

#     return t_hit, np.array([x_hit, y_hit, z_plane], dtype=float)
   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------

# tracker = LinearKalmanFilter(
#     g=9.81,
#     accel_noise_std=0.8,   # tune this
#     meas_noise_std=0.015   # tune this from camera noise
# )

# z_catch = 0.3  # meters

# def on_ball_measurement(ball_position_3d, timestamp_sec):
#     state = tracker.step(ball_position_3d, timestamp_sec)

#     result = intersect_with_z_plane(state, z_catch, g=9.81)
#     if result is None:
#         return None

#     t_hit, catch_point = result
#     return {
#         "state": state,
#         "time_to_catch_plane": t_hit,
#         "catch_point": catch_point



