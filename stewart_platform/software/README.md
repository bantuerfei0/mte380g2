# Stewart platform ball balancer

From the repository root, run:

```sh
cd stewart_platform/software
python -m src
```

In a Python 3.10+ environment on the Pi, install the three application packages with `python -m pip install -r requirements.txt`. They are `smbus2`, `rpi-hardware-pwm`, and the desktop build of `opencv-python`. OpenCV installs NumPy, which the controller also uses. Enable I²C and configure the three hardware PWM channels on the Pi.

OpenCV provides the settings and live windows, camera tracking, and experiment plots. Simulations and CSV metrics also run without opening a window. From the repository root the equivalent launch command is `python -m stewart_platform.software.src`.

## First run

1. In **Rig settings**, enter the three measured marker XY coordinates in centimetres, relative to your desired centre. Set the usable radius for the ball centre. The supplied marker triangle is an example. Servo order is PWM channels **2, 0, 3** on the configured chip. At a 20 ms period, the highest servo angle uses 8% duty (1600 µs) and the lowest uses 12.4% (2480 µs). The configured limits are in increasing pulse-width order. Enter the three verified park pulses in servo order. Use the bracketed list format shown in the form. The servo probe moves each channel to park ± the probe excursion.
2. In **Live balancer**, physically level the plate and click **Capture level**. Lower each positive edge in turn and use **Read tilt** to verify the corresponding downhill axis reads positive. The LSM6DS3TR-C accelerometer uses bus 1, address 0x6A. Keep the plate stationary during calibration.
3. Remove the ball and click **Probe (ball off)**. The application measures both components of tilt change per microsecond for each actuator. It uses the pseudoinverse of that 2 × 3 matrix to command tilt. Servo horn angles and absolute plate height are unmeasured.
4. Click **Mark 3 dots** and click their centres in configured order. Put the ball on the plate, select **Sample ball**, click inside it, then **Save vision**. Verify the reported coordinates of the ball at independent ruler marks and at the largest intended tilt. Three dots give an affine mapping. Perspective and ball height can bias the measurement even when the three dots match perfectly.
5. Choose **Use home** for startup leveling and pulse feedforward, or **Use continuous** to start from park and correct measured tilt throughout balancing. Choose **Centre**, **Set goal**, or **Append goals** (Enter finishes), then **Record / stop** and **Arm / disarm**. Preview can track without powering the servos. Loss of tracking disarms the loop. The red overlay shows the goal, yellow the aim, and green the ball. A waypoint advances after 1 s within 0.5 cm.

The package reads settings, servo response maps and vision calibration beside the software README. Reopening the application reuses those files. A custom `--config` puts the calibration files beside that config. Trial CSVs go under `software/recordings/`.

## Experiments

For a thin hollow sphere rolling without slip, downhill acceleration is approximately 0.6 g times the tilt in radians. With integral gain zero, the controller requests `Kp × (aim − position) − Kd × velocity` on each axis. Increasing velocity feedback brakes the ball earlier. Midpoint aiming recalculates the midpoint on every frame and halves the proportional correction.

The simulation releases the ball from rest 6 cm from centre with a 50 Hz controller, a 70 ms tilt-response time constant, unequal servo gains, a 4° tilt cap and drag of 0.25/s. With Kp = 0.35, increasing Kd from 0.12 to 0.38 reduces simulated overshoot from 1.936 cm to zero and settling time from 3.72 s to 2.52 s. At Kd = 0.38 midpoint aiming takes 5.40 s. In the homing experiment a second tilt bias is introduced at 4 s. Final radial error at 10 s is 3.568 cm with startup homing and 0.007 cm with continuous IMU feedback. These are model outputs.

For bench comparisons, repeat matched releases with one fixed goal, a single armed interval per CSV and identical release position. Compare direct vs midpoint aiming at the same gains, then home vs continuous at the same gains and aiming mode. Overshoot is projected along the initial approach direction. Settling uses radial error within 0.5 cm through the end of the trial for at least one second. Check against independently measured camera accuracy.

```sh
python -m src simulate
python -m src simulate --plot simulated_response.png
python -m src analyze recordings/trial1.csv recordings/trial2.csv --plot measured.png
```

## Source layout

The [simulated response plot](simulated_response.png) shows both experiments.

| Module under src | Responsibility |
|---|---|
| `__main__.py`, `gui.py` | Entry point, settings form, menu |
| `live.py` | Live GUI, control-loop timing, goal editing, graphs and CSV capture |
| `imu.py` | I²C reads, gravity-referenced tilt and stationary sampling |
| `servos.py` | Three hardware PWM outputs, pulse limits and shutdown |
| `calibration.py` | IMU reference capture, servo probing and startup leveling |
| `control.py` | Ball PID, midpoint targeting and tilt-to-pulse feedback |
| `vision.py` | Camera acquisition, marker tracking and ball position in cm |
| `experiments.py` | Simulated trials, recording analysis and response plots |
| `settings.py` | Configuration validation and file paths |
| `kinematics.py` | Original geometric solver, available separately from the calibrated drive |
