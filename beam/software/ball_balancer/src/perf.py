import time
from contextlib import contextmanager


class Profiler:
    """
    Smoothed timings (ms) and rates (Hz) by name, for the stats panel.
    Each name should have a single writer thread; reads from the GUI thread are fine.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.ms: dict[str, float] = {}
        self.hz: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def _ema(self, d: dict, name: str, v: float) -> None:
        d[name] = v if name not in d else d[name] + self.alpha * (v - d[name])

    def record(self, name: str, ms: float) -> None:
        self._ema(self.ms, name, ms)

    @contextmanager
    def time(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - t0) * 1000)

    def tick(self, name: str) -> None:
        now = time.perf_counter()
        last = self._last.get(name)
        if last is not None and now > last:
            self._ema(self.hz, name, 1 / (now - last))
        self._last[name] = now

    def lines(self) -> list[str]:
        rates = "  ".join(f"{k} {v:.0f}Hz" for k, v in list(self.hz.items()))
        times = [f"  {k:<9}{v:6.2f} ms" for k, v in list(self.ms.items())]
        return [rates, *times]


PERF = Profiler()
