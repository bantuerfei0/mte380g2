import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from src.perf import PROFILER


class PositionSensor(ABC):
    """
    Base class for anything that measures the ball's position.

    Positions are normalised to -1..1 per axis (beam ends at +/-1); None means no ball found.

    To add a sensor: subclass this, implement `acquisition_loop()` and `render_view()`,
    and register it in src/impl/__init__.py. Everything else has a working default.
    `acquisition_loop()` runs on its own thread and calls `_publish_measurement()` once per
    measurement, with the time the raw data was acquired.
    """

    num_axes = 1
    view_names: tuple[str, ...] = ("Default",)  # what render_view() can show; cycled with the View button

    def __init__(self, config: dict, calibration: dict) -> None:
        self._measurement_ready = threading.Condition()
        self._measurement_count = 0
        self._last_read_count = 0
        self._latest_position: np.ndarray | None = None
        self._latest_timestamp = 0.0
        self.stop_requested = threading.Event()
        self._thread = threading.Thread(target=self.acquisition_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Subclasses should call this, then release their hardware."""
        self.stop_requested.set()
        self._thread.join(timeout=1)

    def _publish_measurement(self, position: np.ndarray | None, acquired_at: float) -> None:
        PROFILER.record_duration_ms("process", (time.perf_counter() - acquired_at) * 1000)
        PROFILER.count_event("sensor")
        with self._measurement_ready:
            self._latest_position, self._latest_timestamp = position, acquired_at
            self._measurement_count += 1
            self._measurement_ready.notify_all()

    def wait_for_measurement(self, timeout: float) -> tuple[np.ndarray | None, float] | None:
        """Blocks until a measurement newer than the last one read arrives. Returns (position, timestamp), or None on timeout."""
        with self._measurement_ready:
            arrived = self._measurement_ready.wait_for(lambda: self._measurement_count != self._last_read_count, timeout)
            if not arrived:
                return None
            self._last_read_count = self._measurement_count
            return self._latest_position, self._latest_timestamp

    # --- to implement ---

    @abstractmethod
    def acquisition_loop(self) -> None:
        """Runs until `stop_requested` is set, calling `_publish_measurement()` for each new measurement."""

    @abstractmethod
    def render_view(self, view_name: str) -> np.ndarray:
        """Returns a BGR image of the given view (one of `view_names`)."""

    # --- optional ---

    def draw_tracking_overlay(self, image: np.ndarray, target: np.ndarray) -> None:
        """Draws tracking markers (beam, ball, target) on an image from render_view(). Toggled by the user."""

    def add_controls(self, ui) -> None:
        """Adds this sensor's buttons/sliders. They are removed automatically when the sensor is switched out."""

    def status_lines(self) -> list[str]:
        """Extra lines for the stats panel."""
        return []

    def calibration_state(self) -> dict:
        """Calibration to save in calibration.json; passed back as `calibration` on the next start."""
        return {}
