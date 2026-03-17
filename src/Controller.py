from dataclasses import dataclass

import numpy as np


@dataclass
class Controller:
    """Simple Cartesian PID controller with lightweight history for visualization."""

    def __init__(
        self,
        kp: np.ndarray | None = None,
        ki: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        integral_limit: float = 0.25,
        step_limit: float = 0.15,
    ) -> None:
        self.kp = np.array([0.8, 0.8, 0.8], dtype=np.float64) if kp is None else np.asarray(kp, dtype=np.float64).reshape(3)
        self.ki = np.array([0.0, 0.0, 0.0], dtype=np.float64) if ki is None else np.asarray(ki, dtype=np.float64).reshape(3)
        self.kd = np.array([0.05, 0.05, 0.05], dtype=np.float64) if kd is None else np.asarray(kd, dtype=np.float64).reshape(3)
        self.integral_limit = float(integral_limit)
        self.step_limit = float(step_limit)

        self.integral_error = np.zeros(3, dtype=np.float64)
        self.prev_error = np.zeros(3, dtype=np.float64)
        self.prev_timestamp = None

        self.t = np.array([], dtype=np.float64)
        self.pos = np.empty((0, 3), dtype=np.float64)
        self.cmd_pos = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.integral_error.fill(0.0)
        self.prev_error.fill(0.0)
        self.prev_timestamp = None
        self.t = np.array([], dtype=np.float64)
        self.pos = np.empty((0, 3), dtype=np.float64)
        self.cmd_pos = np.zeros(3, dtype=np.float64)

    def update(self, timestamp: float, target_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)

        self.t = np.append(self.t, float(timestamp))
        self.pos = np.vstack((self.pos, current_pos.reshape(1, 3)))

        error = target_pos - current_pos

        if self.prev_timestamp is None:
            dt = 0.0
        else:
            dt = max((float(timestamp) - float(self.prev_timestamp)) / 1000.0, 1e-3)

        if dt > 0.0:
            self.integral_error += error * dt
            self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
            derivative = (error - self.prev_error) / dt
        else:
            derivative = np.zeros(3, dtype=np.float64)

        control_delta = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        step_norm = np.linalg.norm(control_delta)
        if step_norm > self.step_limit:
            control_delta *= self.step_limit / step_norm

        self.cmd_pos = current_pos + control_delta
        self.prev_error = error
        self.prev_timestamp = float(timestamp)
        return self.cmd_pos.copy()

    def predict_pos(self, tt: float | np.ndarray) -> np.ndarray:
        if np.size(tt) <= 1:
            return self.cmd_pos.reshape(3, 1)

        tt = np.asarray(tt, dtype=np.float64).reshape(-1)
        return np.tile(self.cmd_pos.reshape(3, 1), (1, tt.size))
