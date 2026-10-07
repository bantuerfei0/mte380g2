"""Control law and target generation. Every signal is an array with one entry per axis."""

import numpy as np


class PID:
    """
    PID controller with the same gains on every axis.

    - The derivative is taken on the measured position rather than the error, so a
      sudden target change doesn't cause a kick, and it is low-pass filtered.
    - The integral is clamped to +/- integral_limit to prevent wind-up.
    - The output is clamped to +/- output_limit (1 = full tilt).
    """

    def __init__(self, num_axes: int, kp: float, ki: float, kd: float,
                 derivative_filter_alpha: float, integral_limit: float, output_limit: float = 1.0) -> None:
        self.num_axes = num_axes
        self.kp, self.ki, self.kd = kp, ki, kd
        self.derivative_filter_alpha = derivative_filter_alpha  # 1 = no filtering, smaller = smoother
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.reset()

    def reset(self) -> None:
        zeros = np.zeros(self.num_axes)
        self.last_terms = (zeros, zeros, zeros)  # (P, I, D) contributions of the last update, for display
        self._integral = zeros.copy()
        self._filtered_derivative = zeros.copy()
        self._previous_position: np.ndarray | None = None
        self._previous_time: float | None = None

    def compute_output(self, position: np.ndarray, target: np.ndarray, timestamp: float) -> np.ndarray:
        error = target - position
        if self._previous_time is not None and timestamp > self._previous_time:
            dt = timestamp - self._previous_time
            self._integral = np.clip(self._integral + error * dt, -self.integral_limit, self.integral_limit)
            raw_derivative = -(position - self._previous_position) / dt
            self._filtered_derivative += self.derivative_filter_alpha * (raw_derivative - self._filtered_derivative)
        self._previous_position, self._previous_time = position, timestamp

        self.last_terms = (self.kp * error, self.ki * self._integral, self.kd * self._filtered_derivative)
        return np.clip(sum(self.last_terms), -self.output_limit, self.output_limit)


class GoalSequence:
    """
    Loops through a list of goals: [{"position": [...], "hold_seconds": s}, ...].

    A goal is reached once the ball is within `tolerance` of it (distance over all axes)
    and has stayed there for `hold_seconds`; leaving the tolerance restarts the hold.
    A single goal is simply a fixed setpoint.
    """

    def __init__(self, goals: list[dict], tolerance: float) -> None:
        self.goals = [(np.array(goal["position"], float), goal["hold_seconds"]) for goal in goals]
        self.tolerance = tolerance
        self.index = 0
        self._entered_tolerance_at: float | None = None

    @property
    def current_target(self) -> np.ndarray:
        return self.goals[self.index][0]

    def update_target(self, position: np.ndarray | None, timestamp: float) -> np.ndarray:
        """Advances to the next goal if the current one has been reached; returns the target to use."""
        goal, hold_seconds = self.goals[self.index]
        within_tolerance = position is not None and np.linalg.norm(position - goal) <= self.tolerance

        if not within_tolerance:
            self._entered_tolerance_at = None
        elif self._entered_tolerance_at is None:
            self._entered_tolerance_at = timestamp
        elif timestamp - self._entered_tolerance_at >= hold_seconds and len(self.goals) > 1:
            self.index = (self.index + 1) % len(self.goals)
            self._entered_tolerance_at = None
        return self.current_target

    def status_text(self, now: float) -> str:
        hold_seconds = self.goals[self.index][1]
        held = "-" if self._entered_tolerance_at is None else f"{now - self._entered_tolerance_at:.1f}/{hold_seconds}s"
        return f"Goal {self.index + 1}/{len(self.goals)}  held {held}"
