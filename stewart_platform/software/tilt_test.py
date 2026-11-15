from imu import IMU
import time

imu = IMU()
try:
    while True:
        print("%8.1f %8.1f %8.1f" % imu.read()[1])
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    imu.close()
