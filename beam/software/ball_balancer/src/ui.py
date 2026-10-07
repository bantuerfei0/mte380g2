from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, DARK, PANEL, ACCENT = (255, 255, 255), (150, 150, 150), (60, 60, 60), (30, 30, 30), (0, 220, 255)


def text(img: np.ndarray, s: str, org: tuple[int, int], color=WHITE, scale: float = 0.45) -> None:
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


@dataclass
class Button:
    label: str | Callable[[], str]  # a callable label updates live, e.g. "Record" / "Stop rec"
    cb: Callable[[], None]
    group: str
    rect: tuple = (0, 0, 0, 0)


@dataclass
class Slider:
    label: str
    obj: object
    attr: str
    lo: float
    hi: float
    group: str
    rect: tuple = (0, 0, 0, 0)

    def set_from_x(self, x: int) -> None:
        x0, _, x1, _ = self.rect
        f = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
        setattr(self.obj, self.attr, self.lo + f * (self.hi - self.lo))


class UI:
    """
    One window: the sensor's image on the left, a control panel on the right.
    Everything is drawn with cv2.putText, so no Qt widgets or fonts are involved.

    Widgets belong to a group so a component can remove its own controls (e.g. on sensor switch).
    Sliders bind directly to an attribute, so the panel always shows the live value.
    Clicks on the image go to the active capture handler (calibration); Enter cancels it.
    """

    W, ROW = 260, 26

    def __init__(self, name: str) -> None:
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(name, self._mouse)
        self.widgets: list[Button | Slider] = []
        self._group = ""
        self._drag: Slider | None = None
        self._click = None
        self._img_w = 0
        self.prompt = ""

    @contextmanager
    def group(self, name: str):
        prev, self._group = self._group, name
        try:
            yield
        finally:
            self._group = prev

    def add_button(self, label, cb) -> None:
        self.widgets.append(Button(label, cb, self._group))

    def add_slider(self, label: str, obj, attr: str, lo: float, hi: float) -> None:
        self.widgets.append(Slider(label, obj, attr, lo, hi, self._group))

    def remove(self, group: str) -> None:
        self.widgets = [w for w in self.widgets if w.group != group]

    def capture_clicks(self, handler, prompt: str) -> None:
        self._click, self.prompt = handler, prompt

    def end_capture(self) -> None:
        self._click, self.prompt = None, ""

    def _mouse(self, event, x, y, flags, _) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for w in self.widgets:
                x0, y0, x1, y1 = w.rect
                if x0 <= x < x1 and y0 <= y < y1:
                    if isinstance(w, Button):
                        w.cb()
                    else:
                        self._drag = w
                        w.set_from_x(x)
                    return
            if self._click and x < self._img_w:
                self._click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self._drag.set_from_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _panel(self, stats: list[str]) -> np.ndarray:
        p = np.full((2000, self.W, 3), PANEL, np.uint8)
        ox, pad, y = self._img_w, 8, 8
        x = pad
        for b in (w for w in self.widgets if isinstance(w, Button)):
            label = b.label() if callable(b.label) else b.label
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            bw = tw + 2 * pad
            if x + bw > self.W - pad:
                x, y = pad, y + self.ROW + 4
            cv2.rectangle(p, (x, y), (x + bw, y + self.ROW), DARK, -1)
            cv2.rectangle(p, (x, y), (x + bw, y + self.ROW), GREY, 1)
            text(p, label, (x + pad, y + 17))
            b.rect = (ox + x, y, ox + x + bw, y + self.ROW)
            x += bw + 4
        y += self.ROW + 12
        for s in (w for w in self.widgets if isinstance(w, Slider)):
            v = getattr(s.obj, s.attr)
            text(p, f"{s.label}: {v:.3g}", (pad, y + 12))
            y0, y1 = y + 18, y + 28
            cv2.rectangle(p, (pad, y0), (self.W - pad, y1), DARK, -1)
            f = min(max((v - s.lo) / (s.hi - s.lo), 0.0), 1.0)
            cv2.rectangle(p, (pad, y0), (pad + int(f * (self.W - 2 * pad)), y1), ACCENT, -1)
            s.rect = (ox + pad, y0 - 4, ox + self.W - pad, y1 + 4)
            y += 38
        y += 6
        for line in stats:
            text(p, line, (pad, y + 12))
            y += 18
        return p[:y + 8]

    def show(self, img: np.ndarray, stats: list[str]) -> int:
        """Draws and displays the window. Returns the key pressed (-1 for none, 27 if closed)."""
        self._img_w = img.shape[1]
        if self.prompt:
            h = img.shape[0]
            cv2.rectangle(img, (0, h - 28), (self._img_w, h), (0, 0, 0), -1)
            text(img, self.prompt, (8, h - 9), ACCENT, 0.55)
        panel = self._panel(stats)
        h = max(img.shape[0], panel.shape[0])
        pad = lambda a, c: cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=c)
        cv2.imshow(self.name, np.hstack((pad(img, (0, 0, 0)), pad(panel, PANEL))))
        key = cv2.waitKey(1)
        if key in (10, 13) and self._click:
            self.end_capture()
        if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            return 27
        return key
