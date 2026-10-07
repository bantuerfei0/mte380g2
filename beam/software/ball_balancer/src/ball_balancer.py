import json
import time
from pathlib import Path

import cv2
import numpy as np

from src import impl
from src.controller import PID, Goals
from src.servo import Servo
from src.ui import UI

ROOT = Path(__file__).resolve().parent.parent


def fmt(a: np.ndarray | None) -> str:
    return "--" if a is None else " ".join(f"{x:+.3f}" for x in a)


class BallBalancer:
    """
    Threads: the active sensor, one per servo, and this main thread (control + GUI).
    config.json holds hardware settings; calibration.json is rewritten on exit.
    Keys: s = next sensor, v = next view, q / Esc = quit.
    """

    def __init__(self, cfg_path: Path = ROOT / "config.json", cal_path: Path = ROOT / "calibration.json") -> None:
        self.cfg = json.loads(cfg_path.read_text())
        self.cal_path = cal_path
        self.cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}

        p = {**self.cfg["pid"], **self.cal.get("pid", {})}
        self.pid = PID(1, p["kp"], p["ki"], p["kd"], p["d_alpha"], p["i_limit"])
        self.goals = Goals(self.cfg["goals"])
        levels = self.cal.get("servo_levels", [])
        self.servos = [Servo(c, levels[i] if i < len(levels) else None) for i, c in enumerate(self.cfg["servos"])]

        self.ui = UI("Ball Balancer")
        self.ui.add_button("Sensor", self.next_sensor)
        self.ui.add_button("View", self.next_view)
        for name, mx in zip(("kp", "ki", "kd"), self.cfg["pid"]["slider_max"]):
            self.ui.add_slider(name.capitalize(), self.pid, name, 0, mx)
        for i, s in enumerate(self.servos):
            self.ui.add_slider(f"Servo {i} level %", s, "level", s.duty_min, s.duty_max)

        self.sensor, self.name, self.view, self.message = None, "", 0, ""
        names = list(impl.SENSORS)
        if not self.switch(self.cfg["sensor"]) and not any(self.switch(n) for n in names):
            raise RuntimeError(f"No sensor could be started: {self.message}")

    # --- sensor / view switching ---

    def switch(self, name: str) -> bool:
        try:
            sensor = impl.load(name)(self.cfg[name], self.cal.get(name, {}))
        except Exception as e:
            self.message = f"{name}: {e}"
            return False
        if self.sensor:
            self.cal[self.name] = self.sensor.state()
            self.sensor.stop()
        self.ui.end_capture()
        self.ui.remove("sensor")
        with self.ui.group("sensor"):
            sensor.add_controls(self.ui)
        sensor.start()
        self.sensor, self.name, self.view, self.message = sensor, name, 0, ""
        self.pid.dims = sensor.dims
        self.pid.reset()
        return True

    def next_sensor(self) -> None:
        names = list(impl.SENSORS)
        i = names.index(self.name)
        for n in names[i + 1:] + names[:i]:
            if self.switch(n):
                return

    def next_view(self) -> None:
        self.view = (self.view + 1) % len(self.sensor.views)

    # --- main loop ---

    def run(self) -> None:
        for s in self.servos:
            s.start()
        pos, u, rate, last = None, np.zeros(1), 0.0, None
        target = self.goals.target(time.perf_counter())
        try:
            while True:
                m = self.sensor.read(timeout=0.02)
                if m is not None:
                    pos, t = m
                    if last is not None and t > last:
                        rate += 0.1 * (1 / (t - last) - rate)
                    last = t
                    target = self.goals.target(t)
                    if pos is None:
                        self.pid.reset()
                        u = np.zeros(self.sensor.dims)
                    else:
                        u = self.pid.update(pos, target, t)
                    for s, ui in zip(self.servos, u):
                        s.u = float(ui)
                img = self.sensor.render(self.sensor.views[self.view], target)
                key = self.ui.show(img, self._stats(pos, target, u, rate))
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    self.next_sensor()
                elif key == ord("v"):
                    self.next_view()
        finally:
            for s in self.servos:
                s.stop()
            self.sensor.stop()
            self._save()
            cv2.destroyAllWindows()

    def _stats(self, pos, target, u, rate) -> list[str]:
        p, i, d = self.pid.terms
        lines = [
            f"{self.name} / {self.sensor.views[self.view]}   {rate:.0f} Hz",
            f"Pos     {fmt(pos)}",
            f"Target  {fmt(target)}",
            f"Error   {fmt(None if pos is None else target - pos)}",
            f"P {fmt(p)}  I {fmt(i)}",
            f"D {fmt(d)}  Out {fmt(u)}",
            "Duty %  " + " ".join(f"{s.duty():.2f}" for s in self.servos),
            *self.sensor.stats(),
        ]
        if self.message:
            lines.append(self.message[:40])
        return lines

    def _save(self) -> None:
        self.cal[self.name] = self.sensor.state()
        self.cal["pid"] = {"kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd}
        self.cal["servo_levels"] = [s.level for s in self.servos]
        self.cal_path.write_text(json.dumps(self.cal, indent=2))
