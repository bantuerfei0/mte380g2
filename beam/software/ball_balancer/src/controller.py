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
    """
    Loops through goals [{"pos": [...], "hold": s}, ...]. A goal only counts as reached once the
    ball is within `tol` of it (distance over all axes) and has stayed there for `hold` seconds;
    leaving the tolerance restarts the hold. A single goal is just a fixed setpoint.
    """

    def __init__(self, goals: list[dict], tol: float) -> None:
        self.points = [(np.array(g["pos"], float), g["hold"]) for g in goals]
        self.tol = tol
        self.i = 0
        self._since = None  # time the ball entered the tolerance of the current goal

    @property
    def current(self) -> np.ndarray:
        return self.points[self.i][0]

    def update(self, pos: np.ndarray | None, t: float) -> np.ndarray:
        goal, hold = self.points[self.i]
        if pos is None or np.linalg.norm(pos - goal) > self.tol:
            self._since = None
        elif self._since is None:
            self._since = t
        elif t - self._since >= hold and len(self.points) > 1:
            self.i = (self.i + 1) % len(self.points)
            self._since = None
        return self.current

    def status(self, t: float) -> str:
        held = "-" if self._since is None else f"{t - self._since:.1f}/{self.points[self.i][1]}s"
        return f"Goal {self.i + 1}/{len(self.points)}  held {held}"
