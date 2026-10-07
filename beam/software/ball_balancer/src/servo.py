"""Beam servo driven by the Raspberry Pi's hardware PWM."""

import threading
import time

try:
    from rpi_hardware_pwm import HardwarePWM
except ImportError:  # e.g. developing on a laptop: the GUI still runs, the servo just does nothing
    HardwarePWM = None


class Servo:
    """
    The control loop sets `command` (-1..1, normalised tilt); a background thread writes
    the matching duty cycle to the PWM at `update_rate_hz`.

    `level_duty` is the duty cycle (%) at which the beam is flat. A command of +/-1
    maps to level_duty +/- half of the configured duty range, clamped to that range.
    """

    def __init__(self, config: dict, level_duty: float | None = None) -> None:
        self.duty_min = config["duty_min_percent"]
        self.duty_max = config["duty_max_percent"]
        self.half_duty_range = (self.duty_max - self.duty_min) / 2
        self.level_duty = level_duty if level_duty is not None else self.duty_min + self.half_duty_range
        self.direction = -1.0 if config["invert"] else 1.0
        self.update_period_s = 1 / config["update_rate_hz"]
        self.command = 0.0

        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._pwm_update_loop, daemon=True)
        if HardwarePWM is None:
            print("rpi_hardware_pwm not available, servo output disabled")
            self._pwm = None
        else:
            self._pwm = HardwarePWM(pwm_channel=config["pwm_channel"], hz=config["pwm_frequency_hz"],
                                    chip=config["pwm_chip"])

    def current_duty(self) -> float:
        duty = self.level_duty + self.direction * self.command * self.half_duty_range
        return min(max(duty, self.duty_min), self.duty_max)

    def start(self) -> None:
        if self._pwm:
            self._pwm.start(self.current_duty())
        self._thread.start()

    def stop(self) -> None:
        """Returns the beam to level, then releases the PWM."""
        self._stop_requested.set()
        self._thread.join(timeout=1)
        if self._pwm:
            self._pwm.change_duty_cycle(self.level_duty)
            time.sleep(0.3)  # give the servo time to reach level before the signal stops
            self._pwm.stop()

    def _pwm_update_loop(self) -> None:
        last_written_duty = None
        next_update = time.perf_counter()
        while not self._stop_requested.is_set():
            duty = self.current_duty()
            if duty != last_written_duty and self._pwm:
                self._pwm.change_duty_cycle(duty)
                last_written_duty = duty
            next_update += self.update_period_s
            time.sleep(max(0.0, next_update - time.perf_counter()))
