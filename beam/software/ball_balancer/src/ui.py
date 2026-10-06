import time
from collections import deque

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


class UI:
    """
    Main window: buttons drawn on the image, native trackbars, and click capture for
    calibration (a handler receives image clicks until end_capture(); Enter cancels).
    """

    PAD, H = 6, 24

    def __init__(self, name: str) -> None:
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(name, self._mouse)
        self.buttons = []
        self.sliders: dict[str, float] = {}
        self._x = self.PAD
        self._click = None
        self.prompt = ""

    def add_button(self, label: str, cb) -> None:
        (w, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        rect = (self._x, self.PAD, self._x + w + 2 * self.PAD, self.PAD + self.H)
        self.buttons.append((rect, label, cb))
        self._x = rect[2] + self.PAD

    def add_slider(self, name: str, lo: float, hi: float, step: float, value: float, cb) -> None:
        """Trackbars only show raw integer ticks, so the scaled value is drawn on the image."""
        self.sliders[name] = value

        def on_change(v: int) -> None:
            self.sliders[name] = lo + v * step
            cb(self.sliders[name])

        cv2.createTrackbar(name, self.name, round((value - lo) / step), round((hi - lo) / step), on_change)

    def capture_clicks(self, handler, prompt: str) -> None:
        self._click, self.prompt = handler, prompt

    def end_capture(self) -> None:
        self._click, self.prompt = None, ""

    def _mouse(self, event, x, y, *_) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x0, y0, x1, y1), _, cb in self.buttons:
            if x0 <= x < x1 and y0 <= y < y1:
                cb()
                return
        if self._click:
            self._click(x, y)

    def show(self, img: np.ndarray, status: str = "") -> int:
        for (x0, y0, x1, y1), label, _ in self.buttons:
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
            cv2.putText(img, label, (x0 + self.PAD, y1 - 7), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        w = img.shape[1]
        for i, (name, v) in enumerate(self.sliders.items()):
            label = f"{name}: {v:.3g}"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            org = (w - tw - self.PAD, self.PAD + self.H + 20 + 20 * i)
            cv2.putText(img, label, org, FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, org, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        text = self.prompt or status
        if text:
            cv2.putText(img, text, (self.PAD, img.shape[0] - 10), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.name, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13) and self._click:
            self.end_capture()
        return key


class FpsCounter:
    """True average over the last `window` seconds."""

    def __init__(self, window: float = 1.0) -> None:
        self._window = window
        self._stamps: deque[float] = deque()

    def tick(self) -> float:
        now = time.perf_counter()
        self._stamps.append(now)
        while now - self._stamps[0] > self._window:
            self._stamps.popleft()
        span = now - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0
