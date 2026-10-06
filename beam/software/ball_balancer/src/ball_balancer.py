import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.controller import PID, Goals
from src.servo import Servo
from src.ui import UI, FpsCounter

ROOT = Path(__file__).resolve().parent.parent


class BallBalancer:
    """
    Threads: sensor (camera or serial), one per servo, and this main thread (control + GUI).
    config.json holds hardware settings; calibration.json is written on exit.
    """

    def __init__(self, cfg_path: Path = ROOT / "config.json", cal_path: Path = ROOT / "calibration.json") -> None:
        self.cfg = json.loads(cfg_path.read_text())
        self.cal_path = cal_path
        self.cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}

        self.ui = UI("Ball Balancer")
        self.sensor = self._make_sensor()
        p = {**self.cfg["pid"], **self.cal.get("pid", {})}
        self.pid = PID(self.sensor.dims, p["kp"], p["ki"], p["kd"], p["d_alpha"], p["i_limit"])
        self.goals = Goals(self.cfg["goals"])
        levels = self.cal.get("servo_levels", [])
        self.servos = [Servo(c, levels[i] if i < len(levels) else None) for i, c in enumerate(self.cfg["servos"])]
        self._add_controls()

    def _make_sensor(self):
        kind = self.cfg["sensor"]
        if kind == "camera":
            from src.impl.camera_sensor import CameraSensor
            return CameraSensor(self.cfg["camera"], self.cal.get("camera", {}))
        if kind == "lightbar":
            from src.impl.lightbar_sensor import LightBarSensor
            return LightBarSensor(self.cfg["lightbar"], self.cal.get("lightbar", {}))
        raise ValueError(f"Unknown sensor '{kind}'")

    def _add_controls(self) -> None:
        self.sensor.add_controls(self.ui)
        for name, mx in zip(("kp", "ki", "kd"), self.cfg["pid"]["slider_max"]):
            self.ui.add_slider(name.capitalize() + " (PID)", 0, mx, mx / 500, getattr(self.pid, name),
                               lambda v, n=name: setattr(self.pid, n, v))
        for i, s in enumerate(self.servos):
            self.ui.add_slider(f"Servo {i} level %", s.duty_min, s.duty_max, 0.01, s.level,
                               lambda v, s=s: setattr(s, "level", v))

    def run(self) -> None:
        self.sensor.start()
        for s in self.servos:
            s.start()
        fps = FpsCounter()
        rate = 0.0
        target = self.goals.target(time.perf_counter())
        try:
            while True:
                m = self.sensor.read(timeout=0.02)
                if m is not None:
                    pos, t = m
                    rate = fps.tick()
                    target = self.goals.target(t)
                    if pos is None:
                        self.pid.reset()
                        u = np.zeros(self.sensor.dims)
                    else:
                        u = self.pid.update(pos, target, t)
                    for s, ui in zip(self.servos, u):
                        s.u = float(ui)
                key = self.ui.show(self.sensor.render(target), f"{rate:.0f} Hz")
                if key in (ord("q"), 27):
                    break
        finally:
            for s in self.servos:
                s.stop()
            self.sensor.stop()
            self._save()
            cv2.destroyAllWindows()

    def _save(self) -> None:
        self.cal[self.cfg["sensor"]] = self.sensor.state()
        self.cal["pid"] = {"kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd}
        self.cal["servo_levels"] = [s.level for s in self.servos]
        self.cal_path.write_text(json.dumps(self.cal, indent=2))
