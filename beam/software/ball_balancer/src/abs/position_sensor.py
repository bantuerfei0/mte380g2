import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from src.perf import PERF


class PositionSensor(ABC):
    """
    Tracks a ball. Positions are normalised to -1..1 per axis; None means no ball found.

    To add a sensor: subclass this, implement task() and render(), and register it in
    src/impl/__init__.py. task() runs on its own thread and calls _publish() per measurement,
    with t = the time the raw data was acquired (used for timing and profiling).
    Everything else has a working default.
    """

    dims = 1
    views: tuple[str, ...] = ("Default",)  # render() modes, cycled with the View button

    def __init__(self, cfg: dict, cal: dict) -> None:
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
        PERF.record("process", (time.perf_counter() - t) * 1000)
        PERF.tick("sensor")
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

    @abstractmethod
    def task(self) -> None: ...

    @abstractmethod
    def render(self, view: str) -> np.ndarray:
        """Return a BGR image of the given view."""

    def overlay(self, img: np.ndarray, target: np.ndarray) -> None:
        """Draw tracking markers (beam, ball, target) onto a rendered image. Toggled by the user."""

    def add_controls(self, ui) -> None:
        """Register buttons/sliders. They are removed automatically when the sensor is switched out."""

    def stats(self) -> list[str]:
        """Extra lines for the stats panel."""
        return []

    def state(self) -> dict:
        """Calibration data saved to calibration.json and passed back as `cal` next time."""
        return {}
