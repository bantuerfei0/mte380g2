import json
import time
from pathlib import Path

import cv2
import numpy as np

from src import impl
from src.controller import PID, Goals
from src.perf import PERF
from src.recorder import Recorder
from src.servo import Servo
from src.ui import UI

ROOT = Path(__file__).resolve().parent.parent


def fmt(a: np.ndarray | None) -> str:
    return "--" if a is None else " ".join(f"{x:+.3f}" for x in a)


class BallBalancer:
    """
    Threads: the active sensor, one per servo, and this main thread (control + GUI).
    config.json holds hardware settings; calibration.json is rewritten on exit.
    All signals are arrays with one entry per axis, so 2D only needs a 2D sensor and a second servo.
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

        self.overlay, self.graph = True, False
        self.pos, self.target, self.out = None, np.zeros(1), np.zeros(1)

        # (button label, key, action): buttons and keyboard shortcuts come from this one table
        self.actions = [
            ("Sensor", "s", self.next_sensor),
            ("View", "v", self.next_view),
            ("Overlay", "o", lambda: setattr(self, "overlay", not self.overlay)),
            ("Graph", "g", lambda: setattr(self, "graph", not self.graph)),
            (lambda: "Stop rec" if self.recorder.recording else "Record", "r", lambda: self.recorder.toggle()),
        ]
        self.ui = UI("Ball Balancer")
        for label, _, action in self.actions:
            self.ui.add_button(label, action)
        for name, mx in zip(("kp", "ki", "kd"), self.cfg["pid"]["slider_max"]):
            self.ui.add_slider(name.capitalize(), self.pid, name, 0, mx)
        for i, s in enumerate(self.servos):
            self.ui.add_slider(f"Servo {i} level %", s, "level", s.duty_min, s.duty_max)

        self.sensor, self.recorder, self.name, self.view, self.message = None, None, "", 0, ""
        if not self.switch(self.cfg["sensor"]) and not any(self.switch(n) for n in impl.SENSORS):
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
            self.recorder.stop()
        self.ui.end_capture()
        self.ui.remove("sensor")
        with self.ui.group("sensor"):
            sensor.add_controls(self.ui)
        sensor.start()
        self.sensor, self.name, self.view, self.message = sensor, name, 0, ""
        self.recorder = Recorder(sensor.dims, self.cfg["plot_seconds"], ROOT / "recordings")
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
        self.target = self.goals.target(time.perf_counter())
        keys = {ord(k): action for _, k, action in self.actions}
        try:
            while True:
                with PERF.time("wait"):
                    m = self.sensor.read(timeout=0.02)
                if m is not None:
                    self.update(*m)
                with PERF.time("render"):
                    img = self.render()
                with PERF.time("gui"):
                    key = self.ui.show(img, self.stats())
                PERF.tick("loop")
                if key in (ord("q"), 27):
                    break
                if key in keys:
                    keys[key]()
        finally:
            for s in self.servos:
                s.stop()
            self.sensor.stop()
            self.recorder.stop()
            self.save()
            cv2.destroyAllWindows()

    def update(self, pos: np.ndarray | None, t: float) -> None:
        with PERF.time("control"):
            self.target = self.goals.target(t)
            if pos is None:
                self.pid.reset()
                out = np.zeros(self.sensor.dims)
            else:
                out = self.pid.update(pos, self.target, t)
            for s, u in zip(self.servos, out):
                s.u = float(u)
        PERF.record("latency", (time.perf_counter() - t) * 1000)  # acquisition -> servo command
        self.pos, self.out = pos, out
        self.recorder.add(t, self.target, pos, out, self.pid.terms)

    def render(self) -> np.ndarray:
        img = self.sensor.render(self.sensor.views[self.view])
        if self.overlay:
            self.sensor.overlay(img, self.target)
        if self.graph:
            img = np.vstack((img, self.recorder.plot(img.shape[1])))
        return img

    def stats(self) -> list[str]:
        p, i, d = self.pid.terms
        rec = f"REC {self.recorder.path.name}" if self.recorder.recording else ""
        lines = [
            f"{self.name} / {self.sensor.views[self.view]}   {rec}",
            f"Pos     {fmt(self.pos)}",
            f"Target  {fmt(self.target)}",
            f"Error   {fmt(None if self.pos is None else self.target - self.pos)}",
            f"P {fmt(p)}  I {fmt(i)}",
            f"D {fmt(d)}  Out {fmt(self.out)}",
            "Duty %  " + " ".join(f"{s.duty():.2f}" for s in self.servos),
            *self.sensor.stats(),
            "",
            *PERF.lines(),
            "",
            "Keys: " + " ".join(k for _, k, _ in self.actions) + " q",
        ]
        if self.message:
            lines.append(self.message[:40])
        return lines

    def save(self) -> None:
        self.cal[self.name] = self.sensor.state()
        self.cal["pid"] = {"kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd}
        self.cal["servo_levels"] = [s.level for s in self.servos]
        self.cal_path.write_text(json.dumps(self.cal, indent=2))
