"""Three 50 Hz servo outputs, commanded in microseconds."""
from contextlib import contextmanager, ExitStack
import time

import numpy as np

MIN_US, MAX_US = 1600, 2480


@contextmanager
def open_servos(park, chip):
    from rpi_hardware_pwm import HardwarePWM

    with ExitStack() as outputs:
        channels = []
        for channel, pulse in zip((2, 0, 3), park):
            pwm = HardwarePWM(pwm_channel=channel, hz=50, chip=chip)
            pwm.start(float(pulse) / 200)
            outputs.callback(pwm.change_duty_cycle, 12.4)
            channels.append(pwm)

        def write(pulses):
            for pwm, pulse in zip(channels, np.clip(pulses, MIN_US, MAX_US)):
                pwm.change_duty_cycle(float(pulse) / 200)

        yield write


def probe_servos(write, read_tilt, park, step):
    responses = []
    try:
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = step
            write(park - delta)
            time.sleep(1)
            low = read_tilt()
            write(park + delta)
            time.sleep(1)
            change = read_tilt() - low
            print(f"Servo {index + 1}: tilt change {change.round(2)} deg", flush=True)
            responses.append(change / (2 * step))
    finally:
        write(park)
    response = np.column_stack(responses)
    strongest, weakest = np.linalg.svd(response, compute_uv=False)
    if weakest * 2 * step < .3 or strongest > 10 * weakest:
        raise RuntimeError("Probe could not measure independent tilt in both directions")
    return np.linalg.pinv(response)
