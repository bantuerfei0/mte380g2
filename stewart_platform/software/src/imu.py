"""Accelerometer tilt from the plate-mounted LSM6DS3TR-C."""
import struct
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def open_imu(axes):
    from smbus2 import SMBus

    with SMBus(1) as bus:
        address = 0x6A
        if bus.read_byte_data(address, 0x0F) != 0x6A:
            raise RuntimeError("Unexpected IMU device ID")
        bus.write_byte_data(address, 0x12, 0x44)  # Block data update, auto-increment
        bus.write_byte_data(address, 0x17, 0x80)  # Accelerometer low-pass, 8.32 Hz
        bus.write_byte_data(address, 0x10, 0x68)  # 416 Hz, +/-4 g
        time.sleep(0.15)

        def read_tilt():
            axes_samples = []
            for _ in range(30):
                raw = bus.read_i2c_block_data(address, 0x28, 6)
                acceleration = np.array(struct.unpack("<hhh", bytes(raw))) * .122
                axes_samples.append(acceleration)
                time.sleep(.01)
            average = np.mean(axes_samples, axis=0)
            magnitude = np.linalg.norm(average)
            if not 800 < magnitude < 1200:
                raise RuntimeError("IMU average outside the expected gravity range")
            components = average[np.abs(axes) - 1] * np.sign(axes)
            return np.degrees(np.arcsin(np.clip(components / magnitude, -1, 1)))

        yield read_tilt
