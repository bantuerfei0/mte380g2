"""Gravity-referenced plate tilt from the LSM6DS3TR-C accelerometer."""
import struct
import time

import numpy as np


class IMU:
    address = 0x6A

    def __init__(self, config):
        from smbus2 import SMBus

        self.axes = np.array(config["imu_downhill_axes"])
        self.offset = np.array(config["imu_level_offset_deg"])
        self.bus = SMBus(1)
        try:
            if self.bus.read_byte_data(self.address, 0x0F) != 0x6A:
                raise OSError("Unexpected IMU device ID")
            self.bus.write_byte_data(self.address, 0x12, 1)
            deadline = time.monotonic() + .2
            while self.bus.read_byte_data(self.address, 0x12) & 1:
                if time.monotonic() > deadline:
                    raise OSError("IMU reset timed out")
                time.sleep(.002)
            self.bus.write_byte_data(self.address, 0x12, 0x44)  # coherent samples, increment address
            self.bus.write_byte_data(self.address, 0x17, 0x80)  # LPF2, low latency, 416 Hz / 50
            self.bus.write_byte_data(self.address, 0x10, 0x68)  # 416 Hz, +/-4 g
            time.sleep(.15)  # let the accelerometer filter settle before reading tilt
        except BaseException:
            self.close()
            raise

    def tilt(self):
        raw = bytes(self.bus.read_i2c_block_data(self.address, 0x28, 6))
        acceleration = np.array(struct.unpack("<hhh", raw)) * .122
        magnitude = np.linalg.norm(acceleration)
        if not 800 < magnitude < 1200:
            raise ValueError("IMU reading outside stationary gravity range.")
        components = acceleration[abs(self.axes)-1] * np.sign(self.axes)
        return np.degrees(np.arcsin(np.clip(components/magnitude, -1, 1))) - self.offset

    def average(self, samples=20, *, measurement="IMU measurement", stability_limit_deg=.15):
        angles = []
        for _ in range(samples):
            angles.append(self.tilt())
            time.sleep(.01)
        spread = np.std(angles, axis=0)
        if np.max(spread) > stability_limit_deg:
            raise ValueError(
                f"{measurement}: unstable tilt (X {spread[0]:.2f}, "
                f"Y {spread[1]:.2f} deg std dev, limit {stability_limit_deg:.2f})."
            )
        return np.mean(angles, axis=0)

    def close(self):
        self.bus.close()
