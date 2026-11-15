"""Records the system's response for the live plot and for CSV files."""

import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.ui import COLOR_BALL, COLOR_COMMAND, COLOR_TARGET, draw_text

# Recorded signals, each with one CSV column per axis (e.g. target0, position0, ...). All normalised to -1..1.
# "command" is the controller output sent to the servo (0 = level, +/-1 = full tilt).
RECORDED_SIGNALS = ("target", "position", "command", "P", "I", "D")
PLOTTED_SIGNALS = {"target": COLOR_TARGET, "position": COLOR_BALL, "command": COLOR_COMMAND}
PLOT_STRIP_HEIGHT = 160  # pixels per axis


class ResponseRecorder:
    """
    Keeps the last `plot_window_seconds` of samples for the live plot and, while recording,
    writes every sample to recordings/<timestamp>.csv. Position is NaN while the ball is lost.
    """

    def __init__(self, num_axes: int, plot_window_seconds: float, recordings_folder: Path) -> None:
        self.num_axes = num_axes
        self.plot_window_seconds = plot_window_seconds
        self.recordings_folder = recordings_folder
        self.recent_rows: deque[np.ndarray] = deque()  # [timestamp, then each signal's per-axis values]
        self.file_path: Path | None = None
        self._file = None
        self._csv_writer = None
        self._recording_started_at = 0.0

    def column_index(self, signal: str, axis: int) -> int:
        return 1 + RECORDED_SIGNALS.index(signal) * self.num_axes + axis

    def add_sample(self, timestamp: float, target, position, command, pid_terms) -> None:
        if position is None:
            position = np.full(self.num_axes, np.nan)
        row = np.concatenate(([timestamp], target, position, command, *pid_terms))

        self.recent_rows.append(row)
        while timestamp - self.recent_rows[0][0] > self.plot_window_seconds:
            self.recent_rows.popleft()

        if self._csv_writer:
            elapsed = row[0] - self._recording_started_at
            self._csv_writer.writerow([f"{elapsed:.4f}", *(f"{value:.5f}" for value in row[1:])])

    # --- CSV recording ---

    @property
    def is_recording(self) -> bool:
        return self._csv_writer is not None

    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        self.recordings_folder.mkdir(exist_ok=True)
        self.file_path = self.recordings_folder / time.strftime("%Y%m%d-%H%M%S.csv")
        self._file = self.file_path.open("w", newline="")
        self._csv_writer = csv.writer(self._file)
        header = [f"{signal}{axis}" for signal in RECORDED_SIGNALS for axis in range(self.num_axes)]
        self._csv_writer.writerow(["t", *header])
        self._recording_started_at = time.perf_counter()

    def stop_recording(self) -> None:
        if self._file:
            self._file.close()
        self._file = self._csv_writer = None

    # --- live plot ---

    def render_plot(self, width: int) -> np.ndarray:
        """One strip per axis showing target, position and command over the plot window."""
        image = np.zeros((PLOT_STRIP_HEIGHT * self.num_axes, width, 3), np.uint8)
        if len(self.recent_rows) < 2:
            return image

        rows = np.array(self.recent_rows)
        seconds_ago = rows[-1, 0] - rows[:, 0]
        x_pixels = (1 - seconds_ago / self.plot_window_seconds) * (width - 1)

        for axis in range(self.num_axes):
            strip_top = axis * PLOT_STRIP_HEIGHT

            def value_to_y(values):
                return strip_top + (1 - np.clip(values, -1, 1)) / 2 * (PLOT_STRIP_HEIGHT - 1)

            for gridline, color in ((0, (90, 90, 90)), (0.5, (45, 45, 45)), (-0.5, (45, 45, 45))):
                y = int(value_to_y(gridline))
                cv2.line(image, (0, y), (width, y), color, 1)

            for signal, color in PLOTTED_SIGNALS.items():
                points = np.stack((x_pixels, value_to_y(rows[:, self.column_index(signal, axis)])), axis=1)
                gaps = np.flatnonzero(~np.isfinite(points[:, 1]))  # where the ball was lost
                for segment in np.split(points, gaps):
                    segment = segment[np.isfinite(segment[:, 1])]
                    if len(segment) > 1:
                        cv2.polylines(image, [segment.astype(np.int32)], False, color, 1, cv2.LINE_AA)

            draw_text(image, f"axis {axis}", (6, strip_top + 16))
            for i, (signal, color) in enumerate(PLOTTED_SIGNALS.items()):
                draw_text(image, signal, (70 + 80 * i, strip_top + 16), color)
            cv2.line(image, (0, strip_top), (width, strip_top), (120, 120, 120), 1)
        return image
