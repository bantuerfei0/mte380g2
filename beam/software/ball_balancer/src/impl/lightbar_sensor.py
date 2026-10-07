import time

import cv2
import numpy as np
import serial

from src.abs.position_sensor import PositionSensor


class LightBarSensor(PositionSensor):
    """
    Line of light sensors read from the Mega over serial.

    Frame format:
        byte 1: 1 iii 000 v   (MSB=1, 3-bit index, top bit of value)
        byte 2: 0 vvvvvvv     (MSB=0, low 7 bits of value)

    A position is published after each full sweep (index n-1): the weighted centroid of
    each sensor's deviation from its no-ball baseline.
    """

    def __init__(self, cfg: dict, cal: dict) -> None:
        super().__init__()
        self.ser = serial.Serial(cfg["port"], cfg["baud"], timeout=0.1)
        self.n = cfg["num_sensors"]
        self.threshold = cfg["threshold"]
        self.cal_time = cfg["cal_time"]
        self.xs = np.linspace(-1, 1, self.n)
        self.baseline = np.array(cal["baseline"], float) if "baseline" in cal else None
        self.values = np.zeros(self.n)
        self._view = (self.values.copy(), None)
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
        pos = self._locate(v)
        self._view = (v, pos)
        self._publish(pos, t)

    def _locate(self, v: np.ndarray) -> np.ndarray | None:
        if self.baseline is None:
            return None
        w = np.clip(np.abs(self.baseline - v) - self.threshold, 0, None)
        s = w.sum()
        return np.array([w @ self.xs / s]) if s > 0 else None

    def stop(self) -> None:
        super().stop()
        self.ser.close()

    # --- GUI ---

    def add_controls(self, ui) -> None:
        ui.add_button("Calibrate", self._calibrate)
        ui.add_slider("Light threshold", 0, 128, 1, self.threshold, lambda v: setattr(self, "threshold", v))

    def _calibrate(self) -> None:
        self._cal_acc = []
        self._cal_end = time.perf_counter() + self.cal_time

    def render(self, target: np.ndarray) -> np.ndarray:
        v, pos = self._view
        h, w = 240, 640
        img = np.zeros((h, w, 3), np.uint8)
        bw = w // self.n
        top, bot = 40, h - 40
        y = lambda val: int(bot - val / 255 * (bot - top))
        for i, val in enumerate(v):
            x = i * bw
            cv2.rectangle(img, (x + 4, y(val)), (x + bw - 4, bot), (180, 180, 180), -1)
            if self.baseline is not None:
                yb = y(self.baseline[i])
                cv2.line(img, (x + 4, yb), (x + bw - 4, yb), (255, 255, 0), 2)
        px = lambda p: int((p + 1) / 2 * (w - bw) + bw / 2)
        cv2.drawMarker(img, (px(target[0]), h - 20), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if pos is not None:
            cv2.circle(img, (px(pos[0]), h - 20), 8, (0, 255, 0), 2)
        if self._cal_end:
            cv2.putText(img, "Calibrating - remove ball", (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return img

    def state(self) -> dict:
        return {"baseline": self.baseline.tolist()} if self.baseline is not None else {}
