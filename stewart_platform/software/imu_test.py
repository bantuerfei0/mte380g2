import math, time
from imu import IMU

imu = IMU()
try:
    # 1) Static check: board flat and still
    samples = [imu.read()[1] for _ in range(100)]
    for t in samples[:3]:
        print("ax=%.1f ay=%.1f az=%.1f mg" % t)
    mags = [math.sqrt(x*x + y*y + z*z) for x, y, z in samples]
    print("mean |a| = %.1f mg (expect ~1000)" % (sum(mags) / len(mags)))

    # 2) Rate and jitter at 50 Hz
    period, n = 1 / 50, 500
    next_t = time.monotonic()
    stamps = []
    for _ in range(n):
        stamps.append(imu.read()[0])
        next_t += period
        time.sleep(max(0, next_t - time.monotonic()))
    dts = [b - a for a, b in zip(stamps, stamps[1:])]
    print("rate = %.2f Hz, dt min/max = %.1f / %.1f ms"
          % (1 / (sum(dts) / len(dts)), min(dts) * 1e3, max(dts) * 1e3))
finally:
    imu.close()
