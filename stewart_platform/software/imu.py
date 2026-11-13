import ctypes, time

_lib = ctypes.CDLL("./libimu.so")

class IMU:
    def __init__(self):
        if _lib.imu_open() != 0:
            raise RuntimeError("IMU init failed")
        time.sleep(0.01)            # let the first sample arrive

    def read(self):
        """Returns (timestamp_s, (ax, ay, az) in mg), the newest sample."""
        buf = (ctypes.c_float * 3)()
        t = time.monotonic()
        if _lib.imu_read_mg(buf) != 0:
            raise IOError("IMU read failed")
        return t, tuple(buf)

    def close(self):
        _lib.imu_close()
