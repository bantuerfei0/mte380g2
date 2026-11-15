import time
from typing import NamedTuple

import cv2
import numpy as np

from src.abs.position_sensor import PositionSensor
from src.ui import COLOR_BALL, COLOR_REFERENCE, COLOR_TARGET

HSV_UPPER_LIMITS = np.array([179, 255, 255])  # OpenCV's hue range is 0..179


def to_pixel(point) -> tuple[int, int]:
    return int(point[0]), int(point[1])


class CameraSnapshot(NamedTuple):
    """Latest results from the camera thread, replaced as a whole so the GUI never sees a half-updated set."""

    frame: np.ndarray | None
    ball_mask: np.ndarray | None     # thresholded pixels, full frame size
    ball_pixel: np.ndarray | None    # (x, y) centroid, None if no ball
    blob_area_px: float


class CameraSensor(PositionSensor):
    """
    Finds the ball by colour: HSV threshold -> small opening to remove speckle -> image moments.
    Only a region around the beam is processed. The ball's centroid is projected onto the
    line between the two calibrated beam endpoints to give a position from -1 (left) to +1 (right).

    Calibration (saved): the beam endpoints (clicked at startup) and HSV samples of the ball
    (clicked with the Sample button); the threshold is the mean sample +/- the tolerances.
    """

    view_names = ("Camera", "Mask")

    def __init__(self, config: dict, calibration: dict) -> None:
        super().__init__(config, calibration)
        capture_settings = [
            cv2.CAP_PROP_FRAME_WIDTH, config["width"],
            cv2.CAP_PROP_FRAME_HEIGHT, config["height"],
            cv2.CAP_PROP_FPS, config["fps"],
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.CAP_PROP_AUTO_WB, 0,
            cv2.CAP_PROP_AUTO_EXPOSURE, 1,  # 1 = manual exposure on V4L2
            cv2.CAP_PROP_GAIN, 0,
            cv2.CAP_PROP_EXPOSURE, config["exposure"],
        ]
        self.capture = cv2.VideoCapture(config["device_index"], cv2.CAP_V4L2, capture_settings)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open camera {config['device_index']}")
        self.frame_size = np.array([self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)], int)
        self.sample_radius_px = config["sample_patch_px"] // 2
        self.min_blob_area_px = config["min_blob_area_px"]
        self.roi_padding_px = config["roi_padding_px"]
        self.speckle_kernel = np.ones((3, 3), np.uint8)

        # calibration (keys kept stable so existing calibration.json files still load)
        self.hsv_samples: list[list[float]] = calibration.get("samples", [])
        self.hsv_center = np.mean(self.hsv_samples, axis=0) if self.hsv_samples else None
        self.hue_tolerance, self.sat_val_tolerance = calibration.get("tol", [10, 60])
        self.beam_endpoints = np.array(calibration["beam"], float) if "beam" in calibration else None

        self.ui = None
        self._pending_beam_clicks: list[tuple[int, int]] = []
        self._latest = CameraSnapshot(None, None, None, 0.0)
        self._displayed = self._latest  # the snapshot render_view() used, so the overlay matches it

    # --- detection (camera thread) ---

    def _detect_ball(self, frame: np.ndarray) -> tuple[np.ndarray | None, CameraSnapshot]:
        """Returns (normalised position or None, snapshot for display)."""
        hsv_center, beam = self.hsv_center, self.beam_endpoints  # read once; the GUI may replace them
        if hsv_center is None or beam is None:
            return None, CameraSnapshot(frame, None, None, 0.0)

        left_end, right_end = beam
        x0, y0 = np.maximum(np.minimum(left_end, right_end) - self.roi_padding_px, 0).astype(int)
        x1, y1 = np.minimum(np.maximum(left_end, right_end) + self.roi_padding_px, self.frame_size).astype(int)

        tolerance = np.array([self.hue_tolerance, self.sat_val_tolerance, self.sat_val_tolerance])
        lower = np.clip(hsv_center - tolerance, 0, HSV_UPPER_LIMITS).astype(np.uint8)
        upper = np.clip(hsv_center + tolerance, 0, HSV_UPPER_LIMITS).astype(np.uint8)
        roi_hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        roi_mask = cv2.morphologyEx(cv2.inRange(roi_hsv, lower, upper), cv2.MORPH_OPEN, self.speckle_kernel)

        full_mask = np.zeros(frame.shape[:2], np.uint8)
        full_mask[y0:y1, x0:x1] = roi_mask
        moments = cv2.moments(roi_mask, binaryImage=True)
        area = moments["m00"]
        if area < self.min_blob_area_px:
            return None, CameraSnapshot(frame, full_mask, None, area)

        ball_pixel = np.array([moments["m10"] / area + x0, moments["m01"] / area + y0])
        beam_vector = right_end - left_end
        fraction_along_beam = (ball_pixel - left_end) @ beam_vector / (beam_vector @ beam_vector)  # 0..1
        position = np.array([2 * fraction_along_beam - 1])
        return position, CameraSnapshot(frame, full_mask, ball_pixel, area)

    def acquisition_loop(self) -> None:
        while not self.stop_requested.is_set():
            ok, frame = self.capture.read()
            if not ok:
                continue
            acquired_at = time.perf_counter()
            position, self._latest = self._detect_ball(frame)
            self._publish_measurement(position, acquired_at)

    def stop(self) -> None:
        super().stop()
        self.capture.release()

    # --- calibration controls ---

    def add_controls(self, ui) -> None:
        self.ui = ui
        ui.add_button("Beam", self._start_beam_calibration)
        ui.add_button("Sample", lambda: ui.begin_click_capture(self._add_hsv_sample, "Click the ball"))
        ui.add_button("Clear", self._clear_hsv_samples)
        ui.add_slider("Hue tol", self, "hue_tolerance", 0, 90)
        ui.add_slider("Sat/Val tol", self, "sat_val_tolerance", 0, 127)
        self._start_beam_calibration()  # always offered at startup

    def _start_beam_calibration(self) -> None:
        self._pending_beam_clicks = []
        keep_hint = " (Enter = keep saved)" if self.beam_endpoints is not None else ""
        self.ui.begin_click_capture(self._on_beam_click, "Click beam LEFT end" + keep_hint)

    def _on_beam_click(self, x: int, y: int) -> None:
        self._pending_beam_clicks.append((x, y))
        if len(self._pending_beam_clicks) == 1:
            self.ui.prompt_text = "Click beam RIGHT end"
            return
        self.beam_endpoints = np.array(self._pending_beam_clicks, float)
        self._pending_beam_clicks = []
        self.ui.end_click_capture()

    def _add_hsv_sample(self, x: int, y: int) -> None:
        """Averages the HSV colour of a small patch around the click into the ball colour."""
        self.ui.end_click_capture()
        frame = self._latest.frame
        if frame is None:
            return
        r = self.sample_radius_px
        patch = frame[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        self.hsv_samples.append(patch_hsv.mean(axis=0).tolist())
        self.hsv_center = np.mean(self.hsv_samples, axis=0)

    def _clear_hsv_samples(self) -> None:
        self.hsv_samples = []
        self.hsv_center = None

    # --- display ---

    def render_view(self, view_name: str) -> np.ndarray:
        self._displayed = snapshot = self._latest
        if snapshot.frame is None:
            return np.zeros((self.frame_size[1], self.frame_size[0], 3), np.uint8)
        if view_name == "Mask":
            image = (cv2.cvtColor(snapshot.ball_mask, cv2.COLOR_GRAY2BGR) if snapshot.ball_mask is not None
                     else np.zeros_like(snapshot.frame))
        else:
            image = snapshot.frame.copy()
        for click in self._pending_beam_clicks:
            cv2.circle(image, click, 4, COLOR_REFERENCE, -1)
        return image

    def draw_tracking_overlay(self, image: np.ndarray, target: np.ndarray) -> None:
        if self.beam_endpoints is not None:
            left_end, right_end = self.beam_endpoints
            target_pixel = left_end + (target[0] + 1) / 2 * (right_end - left_end)
            cv2.line(image, to_pixel(left_end), to_pixel(right_end), COLOR_REFERENCE, 1)
            cv2.drawMarker(image, to_pixel(target_pixel), COLOR_TARGET, cv2.MARKER_CROSS, 20, 2)
        if self._displayed.ball_pixel is not None:
            cv2.circle(image, to_pixel(self._displayed.ball_pixel), 8, COLOR_BALL, 2)

    def status_lines(self) -> list[str]:
        hsv = "none" if self.hsv_center is None else " ".join(f"{c:.0f}" for c in self.hsv_center)
        return [f"HSV: {hsv} ({len(self.hsv_samples)} samples)",
                f"Blob area: {self._latest.blob_area_px:.0f} px"]

    def calibration_state(self) -> dict:
        state = {"samples": self.hsv_samples, "tol": [self.hue_tolerance, self.sat_val_tolerance]}
        if self.beam_endpoints is not None:
            state["beam"] = self.beam_endpoints.tolist()
        return state
