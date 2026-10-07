"""Lightweight profiling: smoothed section durations and event rates, shown in the stats panel."""

import time
from contextlib import contextmanager

SMOOTHING_ALPHA = 0.05  # weight of each new sample in the moving averages


class Profiler:
    """
    Keeps exponential moving averages of how long named sections take (ms) and how
    often named events happen (Hz). Each name should only be written from one thread;
    reading from the GUI thread is fine.
    """

    def __init__(self) -> None:
        self.durations_ms: dict[str, float] = {}
        self.rates_hz: dict[str, float] = {}
        self._last_event_time: dict[str, float] = {}

    @staticmethod
    def _smooth(averages: dict[str, float], name: str, sample: float) -> None:
        previous = averages.get(name)
        averages[name] = sample if previous is None else previous + SMOOTHING_ALPHA * (sample - previous)

    def record_duration_ms(self, name: str, duration_ms: float) -> None:
        self._smooth(self.durations_ms, name, duration_ms)

    @contextmanager
    def measure(self, name: str):
        """Times the enclosed block: `with PROFILER.measure("render"): ...`"""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_duration_ms(name, (time.perf_counter() - start) * 1000)

    def count_event(self, name: str) -> None:
        """Call once per occurrence; the rate is derived from the time between calls."""
        now = time.perf_counter()
        last = self._last_event_time.get(name)
        if last is not None and now > last:
            self._smooth(self.rates_hz, name, 1 / (now - last))
        self._last_event_time[name] = now

    def summary_lines(self) -> list[str]:
        rates = "  ".join(f"{name} {hz:.0f}Hz" for name, hz in list(self.rates_hz.items()))
        durations = [f"  {name:<9}{ms:6.2f} ms" for name, ms in list(self.durations_ms.items())]
        return [rates, *durations]


# One shared instance so sensors, the controller loop and the GUI all report to the same panel.
PROFILER = Profiler()
