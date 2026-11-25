import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from camera_point_tracking_fuckedupbyjeff import CameraPreview, CameraSensor
from Kinematics_to_PWM_PID import ServoCalibration, StewartPlatformController

# ==========================================
# Configuration
# ==========================================
MARKER_DIST_MM = 70.0         # Dots at (0,0), (0,70), and (70,0) mm
BALL_RADIUS_MM = 20.0         # Standard 40 mm ping-pong ball
CAMERA_HEIGHT_MM = 450.0      # Approximate height from resting platform to camera lens

# Orange ping-pong ball HSV bounds (kept distinct from red marker hue 0..6 / 174..180)
ORANGE_LOWER = np.array([8, 130, 120], dtype=np.uint8)
ORANGE_UPPER = np.array([22, 255, 255], dtype=np.uint8)
KERNEL = np.ones((3, 3), np.uint8)

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


class PlatformCoordinateMapper:
    """
    Maps image pixel coordinates (u, v) to platform-frame (x, y) mm coordinates
    using the 3 tracked surface markers, accounting for foreshortening, induced yaw,
    parasitic center shift, and ball-radius height parallax.

    TLDR the platform tilt will make the platform look stretched/squished from the 
    camera POV, so using Jason's red markers we can undo this and get accurate ball 
    coordinates instead of stretched/squished ones
    """

    def __init__(
        self,
        marker_dist_mm: float = 70.0,       # how far each marker is from the (0,0) marker
        ball_radius_mm: float = 20.0,
        camera_height_mm: float = 450.0,    # what the variable name says
    ):
        self.marker_dist_mm = marker_dist_mm
        self.ball_radius_mm = ball_radius_mm
        self.camera_height_mm = camera_height_mm

        self.calibrated: bool = False
        self.flat_px_per_mm: Optional[float] = None
        self.last_inv_basis: Optional[np.ndarray] = None
        self.last_origin_px: Optional[np.ndarray] = None

    def update_from_markers(self, points: dict, is_flat_ready: bool) -> bool:
        """
        Updates the pixel-to-platform-mm transform from {'top', 'center', 'right'}.
        Returns True if a valid transform is available.
        """
        c = points["center"]
        r = points["right"]
        t = points["top"]

        # Pixel basis vectors corresponding to +X (70 mm) and +Y (70 mm) on the platform
        vec_x = (r - c) / self.marker_dist_mm  # pixels per mm along platform +X
        vec_y = (t - c) / self.marker_dist_mm  # pixels per mm along platform +Y

        # 2x2 Jacobian mapping platform (x, y) mm -> image offset (du, dv) px
        basis = np.column_stack((vec_x, vec_y))
        if abs(np.linalg.det(basis)) < 1e-6:
            return self.last_inv_basis is not None

        self.last_inv_basis = np.linalg.inv(basis)
        self.last_origin_px = c.copy()

        # Lock baseline flat scale once the startup alignment check passes
        if is_flat_ready and not self.calibrated:
            scale_x = float(np.linalg.norm(vec_x))
            scale_y = float(np.linalg.norm(vec_y))
            self.flat_px_per_mm = 0.5 * (scale_x + scale_y)
            self.calibrated = True

        return True

    def pixel_to_platform_mm(
        self,
        uv: Tuple[float, float],
        tilt_alpha_rad: float = 0.0,
        tilt_beta_rad: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Converts ball centroid (u, v) in pixels to platform coordinates (x, y) in mm,
        removing tilt foreshortening and ball-center height parallax.
        """
        if self.last_inv_basis is None or self.last_origin_px is None:
            return None

        delta_px = np.array([uv[0], uv[1]], dtype=float) - self.last_origin_px
        x_mm, y_mm = self.last_inv_basis @ delta_px

        # Correct for parallax caused by the ball's center sitting r_ball above the marker plane
        # When tilted by alpha toward beta, the elevated ball center projects slightly shifted along beta
        if abs(tilt_alpha_rad) > 1e-5:
            parallax_gain = 1.0 + (self.ball_radius_mm / max(100.0, self.camera_height_mm))
            shift_mm = self.ball_radius_mm * math.sin(tilt_alpha_rad) * parallax_gain
            x_mm -= shift_mm * math.cos(tilt_beta_rad)
            y_mm -= shift_mm * math.sin(tilt_beta_rad)

        return float(x_mm), float(y_mm)


def detect_orange_ball(frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Returns (u, v, radius_px) of the orange ping-pong ball, or None."""
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 150:
        return None

    m = cv2.moments(largest)
    if m["m00"] <= 0:
        return None

    u = float(m["m10"] / m["m00"])
    v = float(m["m01"] / m["m00"])
    radius_px = math.sqrt(area / math.pi)
    return u, v, radius_px


def main() -> None:
    sensor = CameraSensor(device_index=0)
    sensor.hue = 110          # Blue marker center hue (OpenCV HSV: 0..179)
    sensor.hue_tolerance = 15 # Detects H in [95, 125], cleanly separated from orange (8..22)
    preview = CameraPreview(sensor)

    mapper = PlatformCoordinateMapper(
        marker_dist_mm=MARKER_DIST_MM,
        ball_radius_mm=BALL_RADIUS_MM,
        camera_height_mm=CAMERA_HEIGHT_MM,
    )

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

    # Command flat level position while waiting for marker lock
    flat_angles = platform.inverse_kinematics(0.0, 0.0)
    flat_pwms = platform.angles_to_pwm(flat_angles)
    # send_to_servos(flat_pwms)

    current_alpha = 0.0
    current_beta = 0.0
    target_pos_mm = (0.0, 0.0)  # Target in platform coordinates (mm)
    prev_time = time.perf_counter()

    try:
        while True:
            frame, mask, points, ready = sensor.read()
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now

            if points is not None:
                mapper.update_from_markers(points, is_flat_ready=ready)

            # Once calibrated on flat platform, begin active ball balancing
            if mapper.calibrated:
                ball_det = detect_orange_ball(frame)
                if ball_det is not None:
                    u, v, rad_px = ball_det
                    ball_mm = mapper.pixel_to_platform_mm(
                        (u, v),
                        tilt_alpha_rad=current_alpha,
                        tilt_beta_rad=current_beta,
                    )

                    if ball_mm is not None:
                        current_alpha, current_beta = platform.compute_pid_tilt(
                            ball_mm, target_pos_mm, dt=dt
                        )
                        angles = platform.inverse_kinematics(current_alpha, current_beta)
                        pwms = platform.angles_to_pwm(angles)
                        # send_to_servos(pwms)

                        # Draw tracked ball overlay on frame before preview renders
                        cv2.circle(frame, (int(round(u)), int(round(v))), int(round(rad_px)), (255, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"Ball: ({ball_mm[0]:.1f}, {ball_mm[1]:.1f}) mm",
                            (int(round(u)) + 12, int(round(v)) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            2,
                        )
                else:
                    # Ball lost/removed -> return smoothly to flat and reset PID state
                    platform.reset_controller()
                    current_alpha, current_beta = 0.0, 0.0
                    # send_to_servos(flat_pwms)

            if preview.show(frame, mask, points, ready):
                break

    finally:
        preview.close()
        sensor.close()


if __name__ == "__main__":
    main()