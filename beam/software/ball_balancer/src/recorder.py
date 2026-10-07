import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.ui import text

# Recorded signals, each with one column per axis. All are in normalised units (-1..1).
FIELDS = ("target", "pos", "out", "P", "I", "D")
COLORS = {"target": (0, 0, 255), "pos": (0, 255, 0), "out": (0, 220, 255)}  # plotted signals


class Recorder:
    """
    Keeps the last `seconds` of the response for the live plot, and writes every
    update to a CSV in `folder` while recording. pos is NaN when the ball is lost.
    """

    def __init__(self, dims: int, seconds: float, folder: Path) -> None:
        self.dims = dims
        self.seconds = seconds
        self.folder = folder
        self.history: deque[np.ndarray] = deque()
        self.path: Path | None = None
        self._file = None
        self._writer = None
        self._t0 = 0.0

    def col(self, field: str, axis: int) -> int:
        return 1 + FIELDS.index(field) * self.dims + axis

    def add(self, t: float, target, pos, out, terms) -> None:
        if pos is None:
            pos = np.full(self.dims, np.nan)
        row = np.concatenate(([t], target, pos, out, *terms))
        self.history.append(row)
        while t - self.history[0][0] > self.seconds:
            self.history.popleft()
        if self._writer:
            self._writer.writerow([f"{row[0] - self._t0:.4f}", *(f"{v:.5f}" for v in row[1:])])

    # --- CSV ---

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def toggle(self) -> None:
        self.stop() if self.recording else self.start()

    def start(self) -> None:
        self.folder.mkdir(exist_ok=True)
        self.path = self.folder / time.strftime("%Y%m%d-%H%M%S.csv")
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t", *(f"{f}{a}" for f in FIELDS for a in range(self.dims))])
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._file:
            self._file.close()
        self._file = self._writer = None

    # --- live plot ---

    def plot(self, width: int, strip_h: int = 160) -> np.ndarray:
        """One strip per axis showing target, pos and out over the last `seconds`."""
        img = np.zeros((strip_h * self.dims, width, 3), np.uint8)
        if len(self.history) < 2:
            return img
        data = np.array(self.history)
        x = ((data[:, 0] - data[-1, 0]) / self.seconds + 1) * (width - 1)
        for a in range(self.dims):
            top = a * strip_h
            y = lambda v: top + (1 - np.clip(v, -1, 1)) / 2 * (strip_h - 1)
            for v, c in ((0, (90, 90, 90)), (0.5, (45, 45, 45)), (-0.5, (45, 45, 45))):
                cv2.line(img, (0, int(y(v))), (width, int(y(v))), c, 1)
            for field, color in COLORS.items():
                pts = np.stack((x, y(data[:, self.col(field, a)])), axis=1)
                ok = np.isfinite(pts[:, 1])
                for seg in np.split(pts, np.flatnonzero(~ok)):  # break the line where the ball was lost
                    seg = seg[np.isfinite(seg[:, 1])]
                    if len(seg) > 1:
                        cv2.polylines(img, [seg.astype(np.int32)], False, color, 1, cv2.LINE_AA)
            text(img, f"axis {a}", (6, top + 16))
            for i, (field, color) in enumerate(COLORS.items()):
                text(img, field, (70 + 60 * i, top + 16), color)
            cv2.line(img, (0, top), (width, top), (120, 120, 120), 1)
        return img
