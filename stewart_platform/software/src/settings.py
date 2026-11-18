"""Rig measurements and session files, relative to the software directory."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "platform_config.json"
RECORDINGS = ROOT / "recordings"
DEFAULTS = dict(camera_index=0, marker_xy_cm=[[0, 10], [8.66, -5], [-8.66, -5]],
                working_radius_cm=10, imu_downhill_axes=[-1, -2], imu_level_offset_deg=[0, 0],
                pwm_chip=0, pwm_channels=[2, 0, 3], park_us=[None]*3, min_us=[None]*3,
                max_us=[None]*3, probe_us=25, max_tilt_deg=4)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate(config, require_pulses=False):
    dots = np.asarray(config["marker_xy_cm"], float)
    axes = config["imu_downhill_axes"]
    if (dots.shape != (3, 2) or not np.isfinite(dots).all()
            or abs(np.linalg.det(dots[1:] - dots[0])) < .1):
        raise ValueError("Enter three noncollinear marker positions in cm.")
    if len(axes) != 2 or any(type(a) != int for a in axes) or set(map(abs, axes)) != {1, 2}:
        raise ValueError("Choose two different signed IMU axes from +/-1, +/-2.")
    offset = np.asarray(config["imu_level_offset_deg"], float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("Enter two finite IMU offsets.")
    channels = config["pwm_channels"]
    if (len(channels) != 3 or len(set(channels)) != 3 or
            any(type(n) != int or n < 0 for n in [*channels, config["pwm_chip"], config["camera_index"]])):
        raise ValueError("Check camera index, PWM chip and three different PWM channels.")
    probe = config["probe_us"]
    if not (0 < config["max_tilt_deg"] <= 15 and 0 < config["working_radius_cm"] < np.inf and 0 < probe < 10000):
        raise ValueError("Check working radius, tilt limit and probe excursion.")
    limits = [config[key] for key in ("min_us", "park_us", "max_us")]
    if any(len(group) != 3 for group in limits):
        raise ValueError("Enter pulse widths for all three servos.")
    for low, park, high in zip(*limits):
        if None in (low, park, high) and not require_pulses:
            continue
        if None in (low, park, high) or not (0 < low <= park-probe and park+probe <= high < 20000):
            raise ValueError("Set each servo's verified limits and park pulse, allowing room for the probe.")


def load(path):
    config = {**DEFAULTS, **(json.loads(path.read_text()) if path.exists() else {})}
    validate(config)
    return config
