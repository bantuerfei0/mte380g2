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

    Calibrate (no ball) records each sensor's baseline b and noise sigma. After each full
    sweep of readings x (index n-1), the position is found in three steps:
      1. d = (x - b) / (255 - b)  shadow as a fraction of each sensor's headroom (0 = no ball)
      2. a sensor sees the ball if d > k * sigma / (255 - b), i.e. k standard deviations above noise
      3. centroid of d over the strongest sensor and its `window` neighbours on each side
    """

    views = ("Readings", "Deviation")

    def __init__(self, cfg: dict, cal: dict) -> None:
        super().__init__(cfg, cal)
        self.ser = serial.Serial(cfg["port"], cfg["baud"], timeout=0.1)
        self.n = cfg["num_sensors"]
        self.k = cfg["k"]
        self.window = cfg["window"]
        self.cal_time = cfg["cal_time"]
        self.xs = np.linspace(-1, 1, self.n)
        self.baseline = np.array(cal["baseline"], float) if "sigma" in cal else None
        self.sigma = np.array(cal["sigma"], float) if "sigma" in cal else None
        self.values = np.zeros(self.n)
        self._view = (self.values.copy(), np.zeros(self.n), np.zeros(self.n), None)  # (readings, d, d threshold, pos)
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
                self.sigma = np.maximum(np.std(self._cal_acc, axis=0), 0.5)  # floor: 8-bit readings can show zero noise
                self._cal_end = 0.0
        d, thr, pos = np.zeros(self.n), np.zeros(self.n), None
        if self.baseline is not None:
            room = np.maximum(255 - self.baseline, 1)
            d = (v - self.baseline) / room
            thr = self.k * self.sigma / room
            pos = self._locate(d, thr)
        self._view = (v, d, thr, pos)
        self._publish(pos, t)

    def _locate(self, d: np.ndarray, thr: np.ndarray) -> np.ndarray | None:
        peak = int(np.argmax(d - thr))
        if d[peak] <= thr[peak]:
            return None
        lo, hi = max(peak - self.window, 0), min(peak + self.window + 1, self.n)
        w = np.clip(d[lo:hi], 0, None)
        return np.array([w @ self.xs[lo:hi] / w.sum()])

    def stop(self) -> None:
        super().stop()
        self.ser.close()

    # --- GUI ---

    def add_controls(self, ui) -> None:
        ui.add_button("Calibrate", self._calibrate)
        ui.add_slider("k (sigma)", self, "k", 0, 10)

    def _calibrate(self) -> None:
        self._cal_acc = []
        self._cal_end = time.perf_counter() + self.cal_time

    H, W = 480, 640

    def _px(self, p: float) -> int:
        bw = self.W // self.n
        return int((p + 1) / 2 * (self.W - bw) + bw / 2)

    def render(self, view: str) -> np.ndarray:
        v, d, thr, _ = self._shown = self._view  # overlay() draws from the same snapshot
        h, w = self.H, self.W
        img = np.zeros((h, w, 3), np.uint8)
        bw = w // self.n
        top, bot = 40, h - 60
        if view == "Readings":  # raw counts, baseline ticks
            bars, ticks, scale, labels = v, self.baseline, 255, [f"{x:.0f}" for x in v]
        else:  # d, with the detection threshold as ticks; auto-scaled
            bars, ticks, labels = d, thr, [f"{x:.2f}" for x in d]
            scale = max(d.max(), thr.max(), 0.02)
        y = lambda val: int(bot - min(max(val, 0) / scale, 1) * (bot - top))
        for i in range(self.n):
            x = i * bw
            cv2.rectangle(img, (x + 4, y(bars[i])), (x + bw - 4, bot), (180, 180, 180), -1)
            text(img, labels[i], (x + 8, bot + 18))
            if ticks is not None:
                cv2.line(img, (x + 4, y(ticks[i])), (x + bw - 4, y(ticks[i])), (255, 255, 0), 2)
        if self._cal_end:
            text(img, "Calibrating - remove ball", (w // 2 - 120, 25), (0, 220, 255), 0.6)
        return img

    def overlay(self, img: np.ndarray, target: np.ndarray) -> None:
        pos = self._shown[3]
        cv2.drawMarker(img, (self._px(target[0]), self.H - 25), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if pos is not None:
            cv2.circle(img, (self._px(pos[0]), self.H - 25), 8, (0, 255, 0), 2)

    def stats(self) -> list[str]:
        return [] if self.baseline is not None else ["Not calibrated"]

    def state(self) -> dict:
        if self.baseline is None:
            return {}
        return {"baseline": self.baseline.tolist(), "sigma": self.sigma.tolist()}
