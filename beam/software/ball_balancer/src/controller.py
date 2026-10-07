import numpy as np


class PID:
    """
    Vectorised PID (same gains on every axis). Derivative is taken on the measurement
    (no kick when the target changes) and low-pass filtered by d_alpha.
    """

    def __init__(self, dims: int, kp: float, ki: float, kd: float,
                 d_alpha: float, i_limit: float, out_limit: float = 1.0) -> None:
        self.dims = dims
        self.kp, self.ki, self.kd = kp, ki, kd
        self.d_alpha = d_alpha
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.reset()

    def reset(self) -> None:
        self.terms = (np.zeros(self.dims),) * 3  # last (P, I, D) contributions, for display
        self._i = np.zeros(self.dims)
        self._d = np.zeros(self.dims)
        self._prev = None
        self._t = None

    def update(self, pos: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
        e = target - pos
        if self._t is not None and t > self._t:
            dt = t - self._t
            self._i = np.clip(self._i + e * dt, -self.i_limit, self.i_limit)
            self._d += self.d_alpha * (-(pos - self._prev) / dt - self._d)
        self._prev, self._t = pos, t
        self.terms = (self.kp * e, self.ki * self._i, self.kd * self._d)
        return np.clip(sum(self.terms), -self.out_limit, self.out_limit)


class Goals:
    """Loops through [(pos, hold_s), ...]. A single entry is just a fixed setpoint."""

    def __init__(self, goals: list[dict]) -> None:
        self.points = [(np.array(g["pos"], float), g["hold"]) for g in goals]
        self.i = 0
        self._t0 = None

    def target(self, t: float) -> np.ndarray:
        if self._t0 is None:
            self._t0 = t
        pos, hold = self.points[self.i]
        if len(self.points) > 1 and t - self._t0 >= hold:
            self.i = (self.i + 1) % len(self.points)
            self._t0 = t
            pos = self.points[self.i][0]
        return pos
