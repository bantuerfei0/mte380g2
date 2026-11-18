"""Measure signed responses of the assembled linkage and establish a level pose."""
from contextlib import ExitStack

import numpy as np

from .control import TiltDrive, inverse_response
from .imu import IMU
from .servos import Servos
from .settings import save, validate
import json


def settled(imu, stop):
    if stop.wait(.45):
        raise RuntimeError("Calibration cancelled.")
    return imu.average()


def identify(servos, imu, probe, stop):
    columns = []
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = probe
        servos.write(servos.park - delta)
        minus = settled(imu, stop)
        servos.write(servos.park + delta)
        plus = settled(imu, stop)
        servos.write(servos.park)
        if np.linalg.norm(plus-minus) < .4:
            raise ValueError(f"Servo {index+1} produced insufficient tilt.")
        columns.append((plus-minus)/(2*probe))
    matrix = np.array(columns).T
    inverse_response(matrix)
    return matrix


def home(servos, inverse, read_settled):
    servos.write(servos.park)
    for _ in range(12):
        tilt = read_settled()
        if np.linalg.norm(tilt) <= .25:
            return servos.pulses.copy()
        servos.write(servos.pulses - .7*(inverse @ tilt))
    raise ValueError("Could not reach level within 0.25 degrees.")


def configure(action, config, path, hardware, stop):
    map_path = path.with_suffix(".response.json")
    with ExitStack() as session:
        if action in ("home", "continuous", "identify"):
            validate(config, require_pulses=True)
        imu = IMU(config)
        session.callback(imu.close)
        if action in ("level", "read"):
            tilt = imu.average(100)
            if action == "read":
                return None, None, f"Tilt X,Y {tilt.round(2)} deg"
            offset = (tilt + imu.offset).tolist()
            save(path, dict(config, imu_level_offset_deg=offset))
            config["imu_level_offset_deg"] = offset
            return None, None, "Level reference saved."
        if action != "identify":
            if not map_path.exists():
                raise ValueError("Remove the ball and probe the servos first.")
            saved = json.loads(map_path.read_text())
            if saved["config"] != config:
                raise ValueError("Configuration changed. Repeat the servo probe.")
            inverse = inverse_response(saved["deg_per_us"])
        servos = Servos(config)
        session.callback(servos.close)
        if action == "identify":
            matrix = identify(servos, imu, config["probe_us"], stop)
            save(map_path, {"config": config, "deg_per_us": matrix.tolist()})
            return None, None, "Servo responses saved."
        rest = home(servos, inverse, lambda: settled(imu, stop)) if action == "home" else servos.park
        drive = TiltDrive(servos, inverse, rest, action == "continuous")
        hardware.enter_context(session.pop_all())
        return imu, drive, "Ready to arm."
