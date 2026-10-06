import time

import cv2
import numpy as np

from src.abs.position_sensor import PositionSensor


def _pt(p) -> tuple[int, int]:
    return int(p[0]), int(p[1])


class CameraSensor(PositionSensor):
    """
    Camera-based ball sensor: HSV threshold + moments inside a ROI around the beam.
    Position is the ball centroid projected onto the calibrated beam line.
    """

    def __init__(self, cfg: dict, cal: dict) -> None:
        super().__init__()
        params = [
            cv2.CAP_PROP_FRAME_WIDTH, cfg["width"],
            cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"],
            cv2.CAP_PROP_FPS, cfg["fps"],
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.CAP_PROP_AUTO_WB, 0,
            cv2.CAP_PROP_AUTO_EXPOSURE, 1,
            cv2.CAP_PROP_GAIN, 0,
            cv2.CAP_PROP_EXPOSURE, cfg["exposure"],
        ]
        self.cam = cv2.VideoCapture(cfg["index"], cv2.CAP_V4L2, params)
        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera {cfg['index']}")
        self.size = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.patch = cfg["patch"] // 2
        self.min_area = cfg["min_area"]
        self.roi_pad = cfg["roi_pad"]
        self.kernel = np.ones((3, 3), np.uint8)

        self.samples: list[list[float]] = cal.get("samples", [])
        self.tol: list[int] = cal.get("tol", [10, 60])
        self.beam = np.array(cal["beam"], float) if "beam" in cal else None
        self.ui = None
        self._pending: list[tuple[int, int]] = []
        self._view = (None, None)  # (frame, ball pixel) swapped as one tuple
        self._params = None        # detection params swapped as one tuple
        self._update_params()

    def _update_params(self) -> None:
        if not self.samples or self.beam is None:
            self._params = None
            return
        c = np.mean(self.samples, axis=0)
        t = np.array([self.tol[0], self.tol[1], self.tol[1]])
        top = np.array([179, 255, 255])
        lo = np.clip(c - t, 0, top).astype(np.uint8)
        hi = np.clip(c + t, 0, top).astype(np.uint8)
        a, b = self.beam
        x0, y0 = np.maximum(np.minimum(a, b) - self.roi_pad, 0).astype(int)
        x1, y1 = np.minimum(np.maximum(a, b) + self.roi_pad, self.size).astype(int)
        ab = b - a
        self._params = (lo, hi, (x0, y0, x1, y1), a, ab, ab @ ab)

    def _detect(self, frame: np.ndarray):
        p = self._params
        if p is None:
            return None, None
        lo, hi, (x0, y0, x1, y1), a, ab, ab2 = p
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, self.kernel)
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] < self.min_area:
            return None, None
        c = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
        return np.array([2 * ((c - a) @ ab) / ab2 - 1]), c

    def task(self) -> None:
        while not self.done.is_set():
            ok, frame = self.cam.read()
            if not ok:
                continue
            t = time.perf_counter()
            pos, ball = self._detect(frame)
            self._view = (frame, ball)
            self._publish(pos, t)

    def stop(self) -> None:
        super().stop()
        self.cam.release()

    # --- GUI ---

    def add_controls(self, ui) -> None:
        self.ui = ui
        ui.add_button("Beam", self._start_beam)
        ui.add_button("Sample", lambda: ui.capture_clicks(self._sample, "Click the ball"))
        ui.add_button("Clear", self._clear)
        ui.add_slider("H tol", 0, 90, 1, self.tol[0], lambda v: self._set_tol(0, v))
        ui.add_slider("SV tol", 0, 127, 1, self.tol[1], lambda v: self._set_tol(1, v))
        self._start_beam()

    def _start_beam(self) -> None:
        self._pending = []
        keep = " (Enter = keep saved)" if self.beam is not None else ""
        self.ui.capture_clicks(self._beam_click, "Click beam LEFT end" + keep)

    def _beam_click(self, x: int, y: int) -> None:
        self._pending.append((x, y))
        if len(self._pending) == 1:
            self.ui.prompt = "Click beam RIGHT end"
            return
        self.beam = np.array(self._pending, float)
        self._pending = []
        self._update_params()
        self.ui.end_capture()

    def _sample(self, x: int, y: int) -> None:
        frame = self._view[0]
        self.ui.end_capture()
        if frame is None:
            return
        r = self.patch
        patch = frame[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        self.samples.append(hsv.mean(axis=0).tolist())
        self._update_params()

    def _clear(self) -> None:
        self.samples = []
        self._update_params()

    def _set_tol(self, i: int, v: float) -> None:
        self.tol[i] = int(v)
        self._update_params()

    def render(self, target: np.ndarray) -> np.ndarray:
        frame, ball = self._view
        if frame is None:
            return np.zeros((self.size[1], self.size[0], 3), np.uint8)
        img = frame.copy()
        for p in self._pending:
            cv2.circle(img, p, 4, (255, 255, 0), -1)
        if self.beam is not None:
            a, b = self.beam
            cv2.line(img, _pt(a), _pt(b), (255, 255, 0), 1)
            cv2.drawMarker(img, _pt(a + (target[0] + 1) / 2 * (b - a)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if ball is not None:
            cv2.circle(img, _pt(ball), 8, (0, 255, 0), 2)
        return img

    def state(self) -> dict:
        s = {"samples": self.samples, "tol": self.tol}
        if self.beam is not None:
            s["beam"] = self.beam.tolist()
        return s
