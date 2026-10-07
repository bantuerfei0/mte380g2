import time

import cv2
import numpy as np

from src.abs.position_sensor import PositionSensor

HSV_MAX = np.array([179, 255, 255])


def _pt(p) -> tuple[int, int]:
    return int(p[0]), int(p[1])


class CameraSensor(PositionSensor):
    """
    HSV threshold + moments inside a ROI around the beam.
    Position is the ball centroid projected onto the calibrated beam line.
    """

    views = ("Camera", "Mask")

    def __init__(self, cfg: dict, cal: dict) -> None:
        super().__init__(cfg, cal)
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
            raise RuntimeError(f"could not open camera {cfg['index']}")
        self.size = np.array([self.cam.get(cv2.CAP_PROP_FRAME_WIDTH), self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)], int)
        self.patch = cfg["patch"] // 2
        self.min_area = cfg["min_area"]
        self.roi_pad = cfg["roi_pad"]
        self.kernel = np.ones((3, 3), np.uint8)

        self.samples: list[list[float]] = cal.get("samples", [])
        self.center = np.mean(self.samples, axis=0) if self.samples else None
        self.h_tol, self.sv_tol = cal.get("tol", [10, 60])
        self.beam = np.array(cal["beam"], float) if "beam" in cal else None
        self.ui = None
        self._pending: list[tuple[int, int]] = []
        self._view = (None, None, None, 0.0)  # (frame, mask, ball px, blob area), swapped as one tuple

    def _detect(self, frame: np.ndarray):
        center, beam = self.center, self.beam
        if center is None or beam is None:
            return None, None, None, 0.0
        a, b = beam
        x0, y0 = np.maximum(np.minimum(a, b) - self.roi_pad, 0).astype(int)
        x1, y1 = np.minimum(np.maximum(a, b) + self.roi_pad, self.size).astype(int)
        tol = np.array([self.h_tol, self.sv_tol, self.sv_tol])
        lo = np.clip(center - tol, 0, HSV_MAX).astype(np.uint8)
        hi = np.clip(center + tol, 0, HSV_MAX).astype(np.uint8)
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        roi = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, self.kernel)
        m = cv2.moments(roi, binaryImage=True)
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[y0:y1, x0:x1] = roi
        if m["m00"] < self.min_area:
            return None, None, mask, m["m00"]
        c = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
        ab = b - a
        return np.array([2 * ((c - a) @ ab) / (ab @ ab) - 1]), c, mask, m["m00"]

    def task(self) -> None:
        while not self.done.is_set():
            ok, frame = self.cam.read()
            if not ok:
                continue
            t = time.perf_counter()
            pos, ball, mask, area = self._detect(frame)
            self._view = (frame, mask, ball, area)
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
        ui.add_slider("Hue tol", self, "h_tol", 0, 90)
        ui.add_slider("Sat/Val tol", self, "sv_tol", 0, 127)
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
        self.ui.end_capture()

    def _sample(self, x: int, y: int) -> None:
        self.ui.end_capture()
        frame = self._view[0]
        if frame is None:
            return
        r = self.patch
        patch = frame[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        self.samples.append(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0).tolist())
        self.center = np.mean(self.samples, axis=0)

    def _clear(self) -> None:
        self.samples = []
        self.center = None

    def render(self, view: str, target: np.ndarray) -> np.ndarray:
        frame, mask, ball, _ = self._view
        if frame is None:
            return np.zeros((self.size[1], self.size[0], 3), np.uint8)
        if view == "Mask":
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask is not None else np.zeros_like(frame)
        else:
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

    def stats(self) -> list[str]:
        center = "none" if self.center is None else " ".join(f"{c:.0f}" for c in self.center)
        return [f"HSV: {center} ({len(self.samples)} samples)", f"Blob area: {self._view[3]:.0f} px"]

    def state(self) -> dict:
        s = {"samples": self.samples, "tol": [self.h_tol, self.sv_tol]}
        if self.beam is not None:
            s["beam"] = self.beam.tolist()
        return s
