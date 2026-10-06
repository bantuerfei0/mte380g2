import threading
import time

try:
    from rpi_hardware_pwm import HardwarePWM
except ImportError:
    HardwarePWM = None


class Servo:
    """
    Main loop sets `u` (-1..1); this thread writes the matching duty to hardware PWM at rate_hz.
    `level` is the duty (%) where the beam is flat; u = +/-1 maps to level +/- half the duty range.
    """

    def __init__(self, cfg: dict, level: float | None = None) -> None:
        self.duty_min, self.duty_max = cfg["duty_min"], cfg["duty_max"]
        self.half = (self.duty_max - self.duty_min) / 2
        self.level = level if level is not None else self.duty_min + self.half
        self.sign = -1.0 if cfg["invert"] else 1.0
        self.period = 1 / cfg["rate_hz"]
        self.u = 0.0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if HardwarePWM is None:
            print("rpi_hardware_pwm not available, servo output disabled")
            self._pwm = None
        else:
            self._pwm = HardwarePWM(pwm_channel=cfg["channel"], hz=cfg["hz"], chip=cfg["chip"])

    def duty(self) -> float:
        d = self.level + self.sign * self.u * self.half
        return min(max(d, self.duty_min), self.duty_max)

    def start(self) -> None:
        if self._pwm:
            self._pwm.start(self.duty())
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=1)
        if self._pwm:
            self._pwm.change_duty_cycle(self.level)
            time.sleep(0.3)
            self._pwm.stop()

    def _run(self) -> None:
        last = None
        nxt = time.perf_counter()
        while not self._done.is_set():
            d = self.duty()
            if d != last and self._pwm:
                self._pwm.change_duty_cycle(d)
                last = d
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))
