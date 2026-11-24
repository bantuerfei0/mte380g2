import time
import cv2
import numpy as np

from Kinematics_to_PWM_PID import ServoCalibration, StewartPlatformController



SERVO_CALIBRATIONS = [
    ServoCalibration(       # channel 2 calibration
        point1=(0.0, 11.1),
        point2=(90.0, 6.9),
        min_pwm=7.5,
        max_pwm=12,
    ),  # Servo 1
    ServoCalibration(       # channel 0 calibration
        point1=(0.0, 11.2),
        point2=(90.0, 6.7),
        min_pwm=7.5,
        max_pwm=12,
    ),  # Servo 2
    ServoCalibration(       # channel 3 calibration
        point1=(0.0, 11.2),
        point2=(90, 6.8),
        min_pwm=7.5,
        max_pwm=12,
    ),  # Servo 3
]


platform = StewartPlatformController(
    R = 140,                # Platform radius (distance from platform centre to ball joints)
    L = 54,                 # Base servo radius (xy-radius from servos to base centre)
    H = 120,                # "Default" height (z-offset from servo acutation point to table)
    l1 = 80,                # Driven link length
    l2 = 94,                # Ball joint link length
    max_tilt_deg=10,        # maximum table tilt
    kp=0.002,
    kd=0.001,
    ki=0.0,
    elbow_up=False,         # Elbows point outward
    calibrations=SERVO_CALIBRATIONS,
)





# 1. Open camera and minimize latency
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Drop stale buffered frames

# Optional: Lock exposure & white balance (values depend on your webcam/OS)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode on many UVC cams
# cap.set(cv2.CAP_PROP_EXPOSURE, -6)

# 2. HSV range for an orange ping-pong ball (tune H/S/V for your room lighting)
ORANGE_LOWER = np.array([5, 130, 120], dtype=np.uint8)
ORANGE_UPPER = np.array([22, 255, 255], dtype=np.uint8)

# Calibration: pixel center of flat platform & mm-per-pixel scale factor
CENTER_PX = (320.0, 240.0)
MM_PER_PX = 0.45  # Measure known platform diameter in pixels to compute this

prev_time = time.perf_counter()

# Define a 3x3 structuring element once before the while loop
KERNEL = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.perf_counter()
    dt = now - prev_time
    prev_time = now

    # Blur slightly to kill sensor noise, then threshold in HSV
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    # Clean up small specular highlights/noise
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 150:  # Ignore tiny false positives
            M = cv2.moments(largest)
            if M["m00"] > 0:
                u = M["m10"] / M["m00"]
                v = M["m01"] / M["m00"]

                # Convert pixel coordinates to platform mm (note: image +v points down)
                ball_x = (u - CENTER_PX[0]) * MM_PER_PX
                ball_y = -(v - CENTER_PX[1]) * MM_PER_PX

                angles, pwms = platform.step((ball_x, ball_y), target_pos=(0.0, 0.0), dt=dt)
                print("Angles:", angles, "PWMs:", pwms)
                # send_to_servos(pwms)

                cv2.circle(frame, (int(u), int(v)), int(np.sqrt(area / np.pi)), (0, 255, 0), 2)
    else:
        # Ball lost / picked up: reset derivative/integral state so it doesn't wind up
        platform.reset_controller()

    cv2.imshow("Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()