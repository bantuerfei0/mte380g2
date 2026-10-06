import threading
from abc import ABC, abstractmethod

import numpy as np


class PositionSensor(ABC):
    """
    Tracks a ball. Positions are normalised to -1..1 per axis; None means no ball found.
    Subclasses run task() on their own thread and call _publish() with each measurement.
    """

    dims = 1

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._seq = 0
        self._last_read = 0
        self._pos: np.ndarray | None = None
        self._t = 0.0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.task, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.done.set()
        self.thread.join(timeout=1)

    def _publish(self, pos: np.ndarray | None, t: float) -> None:
        with self._cond:
            self._pos, self._t = pos, t
            self._seq += 1
            self._cond.notify_all()

    def read(self, timeout: float):
        """Blocks for the next unseen measurement. Returns (pos, t), or None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._last_read, timeout):
                return None
            self._last_read = self._seq
            return self._pos, self._t

    def get_position(self) -> np.ndarray | None:
        with self._cond:
            return self._pos

    @abstractmethod
    def task(self) -> None: ...

    @abstractmethod
    def add_controls(self, ui) -> None:
        """Register buttons/sliders on the main window."""

    @abstractmethod
    def render(self, target: np.ndarray) -> np.ndarray:
        """Return a BGR image for the main window."""

    @abstractmethod
    def state(self) -> dict:
        """Calibration data to save between runs."""
