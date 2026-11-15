import json
import time
from pathlib import Path

import cv2
import numpy as np

from src import impl
from src.controller import PID, GoalSequence
from src.perf import PROFILER
from src.recorder import ResponseRecorder
from src.servo import Servo
from src.ui import KEY_ESCAPE, UI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSOR_WAIT_TIMEOUT_S = 0.02  # keeps the GUI responsive even if the sensor stalls


def format_axes(values: np.ndarray | None) -> str:
    """Formats a per-axis array for the stats panel, e.g. '+0.125 -0.300'."""
    return "--" if values is None else " ".join(f"{v:+.3f}" for v in values)


class BallBalancer:
    """
    Wires a position sensor, the PID controller and the servos together, and runs the GUI.

    Threads: the active sensor's acquisition loop, one PWM loop per servo, and the main
    thread (this class), which runs one control step per new measurement and redraws the window.

    config.json holds hardware settings; calibration.json (tuned values) is rewritten on exit.
    Every signal is an array with one entry per axis, so going 2D only needs a 2D sensor and
    a second servo entry in config.json.
    """

    def __init__(self, config_path: Path = PROJECT_ROOT / "config.json",
                 calibration_path: Path = PROJECT_ROOT / "calibration.json") -> None:
        self.config = json.loads(config_path.read_text())
        self.calibration_path = calibration_path
        self.calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}

        pid_settings = {**self.config["pid"], **self.calibration.get("pid", {})}  # saved gains override defaults
        self.pid = PID(1, pid_settings["kp"], pid_settings["ki"], pid_settings["kd"],
                       pid_settings["derivative_filter_alpha"], pid_settings["integral_limit"])
        self.goals = GoalSequence(self.config["goals"], self.config["goal_tolerance"])
        saved_levels = self.calibration.get("servo_levels", [])
        self.servos = [Servo(servo_config, saved_levels[i] if i < len(saved_levels) else None)
                       for i, servo_config in enumerate(self.config["servos"])]

        self.show_overlay = True
        self.show_graph = False
        self.latest_position: np.ndarray | None = None
        self.current_target = self.goals.current_target
        self.latest_command = np.zeros(1)

        # (button label, keyboard shortcut, action): the panel buttons and the shortcuts both come from this table
        self.actions = [
            ("Sensor", "s", self.switch_to_next_sensor),
            ("View", "v", self.show_next_view),
            ("Overlay", "o", self.toggle_overlay),
            ("Graph", "g", self.toggle_graph),
            (lambda: "Stop rec" if self.recorder.is_recording else "Record", "r",
             lambda: self.recorder.toggle_recording()),
        ]
        self.ui = UI("Ball Balancer")
        for label, _, action in self.actions:
            self.ui.add_button(label, action)
        for gain_name, slider_max in zip(("kp", "ki", "kd"), self.config["pid"]["gain_slider_max"]):
            self.ui.add_slider(gain_name.capitalize(), self.pid, gain_name, 0, slider_max)
        for i, servo in enumerate(self.servos):
            self.ui.add_slider(f"Servo {i} level %", servo, "level_duty", servo.duty_min, servo.duty_max)

        self.sensor = None
        self.sensor_name = ""
        self.view_index = 0
        self.recorder: ResponseRecorder | None = None
        self.last_error_message = ""
        started = self.switch_sensor(self.config["sensor"]) or any(map(self.switch_sensor, impl.SENSOR_CLASSES))
        if not started:
            raise RuntimeError(f"No sensor could be started: {self.last_error_message}")

    # --- sensor and view switching ---

    def switch_sensor(self, name: str) -> bool:
        """Starts sensor `name` in place of the current one. Returns False (and keeps the current one) if it fails."""
        try:
            sensor_class = impl.load_sensor_class(name)
            new_sensor = sensor_class(self.config[name], self.calibration.get(name, {}))
        except Exception as error:  # e.g. camera or serial port not connected
            self.last_error_message = f"{name}: {error}"
            return False

        if self.sensor:
            self.calibration[self.sensor_name] = self.sensor.calibration_state()
            self.sensor.stop()
            self.recorder.stop_recording()
        self.ui.end_click_capture()
        self.ui.remove_group("sensor")
        with self.ui.widget_group("sensor"):
            new_sensor.add_controls(self.ui)
        new_sensor.start()

        self.sensor, self.sensor_name, self.view_index = new_sensor, name, 0
        self.last_error_message = ""
        self.recorder = ResponseRecorder(new_sensor.num_axes, self.config["plot_window_seconds"],
                                         PROJECT_ROOT / "recordings")
        self.pid.num_axes = new_sensor.num_axes
        self.pid.reset()
        self.latest_position, self.latest_command = None, np.zeros(new_sensor.num_axes)
        for servo in self.servos:  # level the beam until the new sensor reports
            servo.command = 0.0
        return True

    def switch_to_next_sensor(self) -> None:
        names = list(impl.SENSOR_CLASSES)
        current = names.index(self.sensor_name)
        for name in names[current + 1:] + names[:current]:
            if self.switch_sensor(name):
                return

    def show_next_view(self) -> None:
        self.view_index = (self.view_index + 1) % len(self.sensor.view_names)

    def toggle_overlay(self) -> None:
        self.show_overlay = not self.show_overlay

    def toggle_graph(self) -> None:
        self.show_graph = not self.show_graph

    # --- main loop ---

    def run(self) -> None:
        for servo in self.servos:
            servo.start()
        shortcuts = {ord(key): action for _, key, action in self.actions}
        try:
            while True:
                with PROFILER.measure("wait"):
                    measurement = self.sensor.wait_for_measurement(SENSOR_WAIT_TIMEOUT_S)
                if measurement is not None:
                    self.run_control_step(*measurement)
                with PROFILER.measure("render"):
                    image = self.compose_image()
                with PROFILER.measure("gui"):
                    key = self.ui.show(image, self.status_lines())
                PROFILER.count_event("loop")

                if key in (ord("q"), KEY_ESCAPE):
                    break
                if key in shortcuts:
                    shortcuts[key]()
        finally:
            for servo in self.servos:
                servo.stop()
            self.sensor.stop()
            self.recorder.stop_recording()
            self.save_calibration()
            cv2.destroyAllWindows()

    def run_control_step(self, position: np.ndarray | None, acquired_at: float) -> None:
        """Updates the target, computes the tilt command and sends it to the servos."""
        with PROFILER.measure("control"):
            self.current_target = self.goals.update_target(position, acquired_at)
            if position is None:  # ball lost: level the beam and start the controller fresh
                self.pid.reset()
                command = np.zeros(self.sensor.num_axes)
            else:
                command = self.pid.compute_output(position, self.current_target, acquired_at)
            for servo, axis_command in zip(self.servos, command):
                servo.command = float(axis_command)
        PROFILER.record_duration_ms("latency", (time.perf_counter() - acquired_at) * 1000)  # sensor -> servo

        self.latest_position, self.latest_command = position, command
        self.recorder.add_sample(acquired_at, self.current_target, position, command, self.pid.last_terms)

    # --- display ---

    def compose_image(self) -> np.ndarray:
        image = self.sensor.render_view(self.sensor.view_names[self.view_index])
        if self.show_overlay:
            self.sensor.draw_tracking_overlay(image, self.current_target)
        if self.show_graph:
            image = np.vstack((image, self.recorder.render_plot(image.shape[1])))
        return image

    def status_lines(self) -> list[str]:
        p_term, i_term, d_term = self.pid.last_terms
        position = self.latest_position
        error = None if position is None else self.current_target - position
        recording = f"REC {self.recorder.file_path.name}" if self.recorder.is_recording else ""
        lines = [
            f"{self.sensor_name} / {self.sensor.view_names[self.view_index]}   {recording}",
            f"Pos     {format_axes(position)}",
            f"Target  {format_axes(self.current_target)}",
            f"        {self.goals.status_text(time.perf_counter())}",
            f"Error   {format_axes(error)}",
            f"P {format_axes(p_term)}  I {format_axes(i_term)}",
            f"D {format_axes(d_term)}  Cmd {format_axes(self.latest_command)}",
            "Duty %  " + " ".join(f"{servo.current_duty():.2f}" for servo in self.servos),
            *self.sensor.status_lines(),
            "",
            *PROFILER.summary_lines(),
            "",
            "Keys: " + " ".join(key for _, key, _ in self.actions) + " q",
        ]
        if self.last_error_message:
            lines.append(self.last_error_message[:40])
        return lines

    def save_calibration(self) -> None:
        self.calibration[self.sensor_name] = self.sensor.calibration_state()
        self.calibration["pid"] = {"kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd}
        self.calibration["servo_levels"] = [servo.level_duty for servo in self.servos]
        self.calibration_path.write_text(json.dumps(self.calibration, indent=2))
