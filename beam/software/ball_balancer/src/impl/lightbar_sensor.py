import time

import cv2
import numpy as np
import serial

from src.abs.position_sensor import PositionSensor
from src.ui import text


class LightBarSensor(PositionSensor):
    """
    Line of light sensors read from the Mega over serial.

    Frame format:
        byte 1: 1 iii 000 v   (MSB=1, 3-bit index, top bit of value)
        byte 2: 0 vvvvvvv     (MSB=0, low 7 bits of value)

    A position is published after each full sweep (index n-1): the weighted centroid of
    each sensor's deviation from its no-ball baseline.
    """

    views = ("Readings", "Deviation")

    def __init__(self, cfg: dict, cal: dict) -> None:
        super().__init__(cfg, cal)
        self.ser = serial.Serial(cfg["port"], cfg["baud"], timeout=0.1)
        self.n = cfg["num_sensors"]
        self.threshold = cfg["threshold"]
        self.cal_time = cfg["cal_time"]
        self.xs = np.linspace(-1, 1, self.n)
        self.baseline = np.array(cal["baseline"], float) if "baseline" in cal else None
        self.values = np.zeros(self.n)
        self._view = (self.values.copy(), np.zeros(self.n), None)  # (readings, deviation, pos)
        self._shown = self._view
        self._cal_acc: list[np.ndarray] = []
        self._cal_end = 0.0

    def task(self) -> None:
        first = None  # holds the first byte of a frame until its partner arrives
        while not self.done.is_set():
            for b in self.ser.read(self.ser.in_waiting or 1):
                if b & 0x80:
                    first = b
                elif first is not None:
                    idx = (first >> 4) & 0x07
                    if idx < self.n:
                        self.values[self.n - 1 - idx] = ((first & 0x01) << 7) | b  # sensor order reversed
                        if idx == self.n - 1:
                            self._sweep()
                    first = None
                # a second byte with no first byte (e.g. right after connecting) is dropped

    def _sweep(self) -> None:
        t = time.perf_counter()
        v = self.values.copy()
        if self._cal_end:
            self._cal_acc.append(v)
            if t >= self._cal_end:
                self.baseline = np.mean(self._cal_acc, axis=0)
                self._cal_end = 0.0
        dev, pos = np.zeros(self.n), None
        if self.baseline is not None:
            dev = np.clip(np.abs(self.baseline - v) - self.threshold, 0, None)
            s = dev.sum()
            pos = np.array([dev @ self.xs / s]) if s > 0 else None
        self._view = (v, dev, pos)
        self._publish(pos, t)

    def stop(self) -> None:
        super().stop()
        self.ser.close()

    # --- GUI ---

    def add_controls(self, ui) -> None:
        ui.add_button("Calibrate", self._calibrate)
        ui.add_slider("Threshold", self, "threshold", 0, 128)

    def _calibrate(self) -> None:
        self._cal_acc = []
        self._cal_end = time.perf_counter() + self.cal_time

    H, W = 480, 640

    def _px(self, p: float) -> int:
        bw = self.W // self.n
        return int((p + 1) / 2 * (self.W - bw) + bw / 2)

    def render(self, view: str) -> np.ndarray:
        v, dev, _ = self._shown = self._view  # overlay() draws from the same snapshot
        h, w = self.H, self.W
        img = np.zeros((h, w, 3), np.uint8)
        bw = w // self.n
        top, bot = 40, h - 60
        y = lambda val: int(bot - min(val, 255) / 255 * (bot - top))
        bars = v if view == "Readings" else dev * 255 / max(dev.max(), 1)
        for i, val in enumerate(bars):
            x = i * bw
            cv2.rectangle(img, (x + 4, y(val)), (x + bw - 4, bot), (180, 180, 180), -1)
            text(img, f"{v[i]:.0f}", (x + 8, bot + 18))
            if view == "Readings" and self.baseline is not None:
                yb = y(self.baseline[i])
                cv2.line(img, (x + 4, yb), (x + bw - 4, yb), (255, 255, 0), 2)
        if self._cal_end:
            text(img, "Calibrating - remove ball", (w // 2 - 120, 25), (0, 220, 255), 0.6)
        return img

    def overlay(self, img: np.ndarray, target: np.ndarray) -> None:
        pos = self._shown[2]
        cv2.drawMarker(img, (self._px(target[0]), self.H - 25), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if pos is not None:
            cv2.circle(img, (self._px(pos[0]), self.H - 25), 8, (0, 255, 0), 2)

    def stats(self) -> list[str]:
        return [] if self.baseline is not None else ["Not calibrated"]

    def state(self) -> dict:
        return {"baseline": self.baseline.tolist()} if self.baseline is not None else {}
