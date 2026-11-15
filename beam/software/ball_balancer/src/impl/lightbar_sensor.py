import time
from typing import NamedTuple

import cv2
import numpy as np
import serial

from src.abs.position_sensor import PositionSensor
from src.ui import ACCENT, COLOR_BALL, COLOR_REFERENCE, COLOR_TARGET, draw_text

ADC_MAX = 255           # sensors report 8-bit values
MIN_NOISE_COUNTS = 0.5  # floor on calibrated noise: 8-bit readings can show zero spread


class LightBarSnapshot(NamedTuple):
    """Results of one sweep, replaced as a whole so the GUI never sees a half-updated set."""

    readings: np.ndarray             # raw ADC counts per sensor
    shadow_fraction: np.ndarray      # step 1 below, per sensor
    detection_threshold: np.ndarray  # step 2 below, per sensor (same units as shadow_fraction)
    position: np.ndarray | None


class LightBarSensor(PositionSensor):
    """
    A row of light sensors under the beam, read from the Arduino Mega over serial.

    Serial frame (2 bytes per sensor reading):
        byte 1: 1 iii 000 v   (MSB=1, 3-bit sensor index, top bit of value)
        byte 2: 0 vvvvvvv     (MSB=0, low 7 bits of value)

    Calibrate (no ball on the beam) records each sensor's baseline and noise (standard
    deviation). After every full sweep of readings, the position is found in three steps:
      1. shadow_fraction = (reading - baseline) / (255 - baseline)
         how much of the sensor's remaining headroom the ball's shadow uses (0 = no ball)
      2. a sensor detects the ball if its shadow_fraction exceeds detection_sigmas noise
         standard deviations (expressed in the same headroom units)
      3. position = centroid of shadow_fraction over the strongest sensor and its
         `centroid_neighbours` neighbours on each side, giving positions between sensors
    """

    view_names = ("Readings", "Shadow")
    IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

    def __init__(self, config: dict, calibration: dict) -> None:
        super().__init__(config, calibration)
        self.serial_port = serial.Serial(config["port"], config["baud"], timeout=0.1)
        self.num_sensors = config["num_sensors"]
        self.detection_sigmas = config["detection_sigmas"]
        self.centroid_neighbours = config["centroid_neighbours"]
        self.calibration_seconds = config["calibration_seconds"]
        self.sensor_positions = np.linspace(-1, 1, self.num_sensors)  # normalised beam position of each sensor

        has_calibration = "sigma" in calibration
        self.baseline = np.array(calibration["baseline"], float) if has_calibration else None
        self.noise_sigma = np.array(calibration["sigma"], float) if has_calibration else None

        self._readings = np.zeros(self.num_sensors)  # filled in one sensor at a time by the serial thread
        self._calibration_sweeps: list[np.ndarray] = []
        self._calibration_ends_at = 0.0  # non-zero while calibrating
        empty = np.zeros(self.num_sensors)
        self._latest = LightBarSnapshot(empty, empty, empty, None)
        self._displayed = self._latest  # the snapshot render_view() used, so the overlay matches it

    # --- serial thread ---

    def acquisition_loop(self) -> None:
        first_byte = None  # held until the second byte of the frame arrives
        while not self.stop_requested.is_set():
            for byte in self.serial_port.read(self.serial_port.in_waiting or 1):
                if byte & 0x80:  # first byte of a frame
                    first_byte = byte
                elif first_byte is not None:
                    sensor_index = (first_byte >> 4) & 0x07
                    if sensor_index < self.num_sensors:
                        value = ((first_byte & 0x01) << 7) | byte
                        # sensors are wired in the opposite order to the beam's -1..+1 direction
                        self._readings[self.num_sensors - 1 - sensor_index] = value
                        if sensor_index == self.num_sensors - 1:  # last sensor of the sweep
                            self._process_sweep()
                    first_byte = None
                # a second byte without a first byte (e.g. right after connecting) is dropped

    def _process_sweep(self) -> None:
        acquired_at = time.perf_counter()
        readings = self._readings.copy()
        if self._calibration_ends_at:
            self._accumulate_calibration(readings, acquired_at)

        position = None
        shadow_fraction = detection_threshold = np.zeros(self.num_sensors)
        if self.baseline is not None:
            headroom = np.maximum(ADC_MAX - self.baseline, 1)
            shadow_fraction = (readings - self.baseline) / headroom
            detection_threshold = self.detection_sigmas * self.noise_sigma / headroom
            position = self._locate_ball(shadow_fraction, detection_threshold)

        self._latest = LightBarSnapshot(readings, shadow_fraction, detection_threshold, position)
        self._publish_measurement(position, acquired_at)

    def _locate_ball(self, shadow_fraction: np.ndarray, detection_threshold: np.ndarray) -> np.ndarray | None:
        strongest = int(np.argmax(shadow_fraction - detection_threshold))
        if shadow_fraction[strongest] <= detection_threshold[strongest]:
            return None  # no sensor clearly sees the ball
        first = max(strongest - self.centroid_neighbours, 0)
        last = min(strongest + self.centroid_neighbours + 1, self.num_sensors)
        weights = np.clip(shadow_fraction[first:last], 0, None)
        return np.array([weights @ self.sensor_positions[first:last] / weights.sum()])

    def _accumulate_calibration(self, readings: np.ndarray, now: float) -> None:
        self._calibration_sweeps.append(readings)
        if now >= self._calibration_ends_at:
            self.baseline = np.mean(self._calibration_sweeps, axis=0)
            self.noise_sigma = np.maximum(np.std(self._calibration_sweeps, axis=0), MIN_NOISE_COUNTS)
            self._calibration_ends_at = 0.0

    def stop(self) -> None:
        super().stop()
        self.serial_port.close()

    # --- controls ---

    def add_controls(self, ui) -> None:
        ui.add_button("Calibrate", self._start_calibration)
        ui.add_slider("Detect (sigmas)", self, "detection_sigmas", 0, 10)

    def _start_calibration(self) -> None:
        self._calibration_sweeps = []
        self._calibration_ends_at = time.perf_counter() + self.calibration_seconds

    # --- display ---

    def _beam_position_to_x(self, position: float) -> int:
        bar_width = self.IMAGE_WIDTH // self.num_sensors
        return int((position + 1) / 2 * (self.IMAGE_WIDTH - bar_width) + bar_width / 2)

    def render_view(self, view_name: str) -> np.ndarray:
        """One bar per sensor. Readings: raw counts with baseline ticks. Shadow: shadow_fraction with threshold ticks."""
        self._displayed = snapshot = self._latest
        width, height = self.IMAGE_WIDTH, self.IMAGE_HEIGHT
        image = np.zeros((height, width, 3), np.uint8)
        bar_width = width // self.num_sensors
        bars_top, bars_bottom = 40, height - 60

        if view_name == "Readings":
            bar_values, tick_values, full_scale = snapshot.readings, self.baseline, ADC_MAX
            labels = [f"{v:.0f}" for v in snapshot.readings]
        else:
            bar_values, tick_values = snapshot.shadow_fraction, snapshot.detection_threshold
            full_scale = max(bar_values.max(), tick_values.max(), 0.02)  # auto-scale to the largest value
            labels = [f"{v:.2f}" for v in snapshot.shadow_fraction]

        def value_to_y(value: float) -> int:
            fraction_of_scale = min(max(value, 0) / full_scale, 1)
            return int(bars_bottom - fraction_of_scale * (bars_bottom - bars_top))

        for i in range(self.num_sensors):
            left, right = i * bar_width + 4, (i + 1) * bar_width - 4
            cv2.rectangle(image, (left, value_to_y(bar_values[i])), (right, bars_bottom), (180, 180, 180), -1)
            draw_text(image, labels[i], (left + 4, bars_bottom + 18))
            if tick_values is not None:
                tick_y = value_to_y(tick_values[i])
                cv2.line(image, (left, tick_y), (right, tick_y), COLOR_REFERENCE, 2)

        if self._calibration_ends_at:
            draw_text(image, "Calibrating - remove ball", (width // 2 - 120, 25), ACCENT, 0.6)
        return image

    def draw_tracking_overlay(self, image: np.ndarray, target: np.ndarray) -> None:
        marker_y = self.IMAGE_HEIGHT - 25
        cv2.drawMarker(image, (self._beam_position_to_x(target[0]), marker_y), COLOR_TARGET, cv2.MARKER_CROSS, 20, 2)
        if self._displayed.position is not None:
            cv2.circle(image, (self._beam_position_to_x(self._displayed.position[0]), marker_y), 8, COLOR_BALL, 2)

    def status_lines(self) -> list[str]:
        return [] if self.baseline is not None else ["Not calibrated - press Calibrate"]

    def calibration_state(self) -> dict:
        if self.baseline is None:
            return {}
        return {"baseline": self.baseline.tolist(), "sigma": self.noise_sigma.tolist()}
