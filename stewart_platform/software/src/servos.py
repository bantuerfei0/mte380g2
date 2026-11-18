"""50 Hz hardware PWM. All commands passed to this module are pulse widths in microseconds."""
from contextlib import ExitStack
import time

import numpy as np

from .settings import validate


class Servos:
    def __init__(self, config):
        from rpi_hardware_pwm import HardwarePWM
        validate(config, require_pulses=True)
        self.low, self.park, self.high = (np.asarray(config[k], float) for k in ("min_us", "park_us", "max_us"))
        self.outputs = []
        self.pulses = self.park.copy()
        self.resources = ExitStack()
        try:
            for channel, pulse in zip(config["pwm_channels"], self.park):
                pwm = HardwarePWM(pwm_channel=channel, hz=50, chip=config["pwm_chip"])
                pwm.start(pulse / 200)
                self.resources.callback(pwm.stop)
                self.outputs.append(pwm)
        except BaseException:
            self.resources.close()
            raise

    def write(self, pulses):
        if np.shape(pulses) != (3,) or not np.isfinite(pulses).all():
            raise ValueError("Expected three finite servo pulse widths.")
        self.pulses = np.clip(pulses, self.low, self.high)
        for pwm, pulse in zip(self.outputs, self.pulses):
            pwm.change_duty_cycle(pulse / 200)

    def close(self):
        try:
            self.write(self.park)
            time.sleep(.3)
        finally:
            self.resources.close()
