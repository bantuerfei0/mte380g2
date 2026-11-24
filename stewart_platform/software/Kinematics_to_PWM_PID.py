from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

"""
For consistency, "platform" refers to the big circular part the ball sits on, and 
    "base" refers to the part with the servos attached. 
"""

"""
This code takes a ball position (ball_pos) and target position (target_pos) as input, and 
    returns three servo angles theta1 (y-axis servo), theta2 (positive x servo), and theta3.

To use, call platform.step(ball_pos, target_pos, dt) once per frame with
    dt = time.perf_counter() - prev_time

Clanker tuning advice: start with ki = 0, adjust kp so a ball near the rim with the centre as the
    target has a platform tilt of 5-8 degrees, then increase kd until oscillations are minimized
"""


@dataclass
class ServoCalibration:
    point1: Tuple[float, float]  # (angle_deg_1, pwm_1)
    point2: Tuple[float, float]  # (angle_deg_2, pwm_2)
    min_pwm: Optional[float] = None
    max_pwm: Optional[float] = None

    def __post_init__(self) -> None:
        a1, p1 = self.point1
        a2, p2 = self.point2

        if abs(a2 - a1) < 1e-9:
            raise ValueError("Calibration points must have two distinct angles.")

        self.slope: float = (p2 - p1) / (a2 - a1)
        self.pwm_at_zero: float = p1 - self.slope * a1

        low, high = min(p1, p2), max(p1, p2)
        self._min_pwm: float = low if self.min_pwm is None else self.min_pwm
        self._max_pwm: float = high if self.max_pwm is None else self.max_pwm

    def angle_to_pwm(self, angle_deg: float) -> float:
        """Converts geometric angle (degrees) to calibrated PWM value."""
        pwm = self.pwm_at_zero + self.slope * angle_deg
        return max(self._min_pwm, min(self._max_pwm, pwm))


class StewartPlatformController:

    def __init__(
        self,
        R: float,   # radius from centre of platform to ball joint (top of linkage)
        L: float,   # length from centre of base to servo actuation point (bottom of linkage)
        H: float,   # z-offset of the platform. Changes the idle servo angle (angle when ball_pos = target_pos)
        l1: float,  # the length of the driven link (i.e. the one connected to the servo)
        l2: float,  # the length of the second link (i.e. the one connected to the ball joint)
        calibrations: Sequence[ServoCalibration],
        max_tilt_deg: float = 15.0,     # if the tilt is too much parallax will fuck up the camera but i dont wanna think about that
        kp: float = 0.005,
        kd: float = 0.003,
        ki: float = 0.0,
        elbow_up: bool = False,     # False for linkage "elbows" flaring outwards (ideally never True)
    ):
        """
        3-DOF Stewart Platform Controller & Calibrated Inverse Kinematics Solver.
        """
        if len(calibrations) != 3:
            raise ValueError(
                "Exactly 3 ServoCalibration objects (for S1, S2, S3) must be provided."
            )

        self.R = R
        self.L = L
        self.H = H
        self.l1 = l1
        self.l2 = l2
        self.calibrations: List[ServoCalibration] = list(calibrations)
        self.max_tilt_rad = math.radians(max_tilt_deg)
        self.elbow_sign = 1.0 if elbow_up else -1.0

        # PID Gains
        self.kp = kp
        self.kd = kd
        self.ki = ki

        # Controller state
        self.prev_ball_pos: Optional[Tuple[float, float]] = None
        self.integral_x: float = 0.0
        self.integral_y: float = 0.0


    
    def reset_controller(self) -> None:

        """Resets velocity estimation and integral accumulators."""

        self.prev_ball_pos = None
        self.integral_x = 0.0
        self.integral_y = 0.0



    def compute_pid_tilt(
        self,
        ball_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        dt: float = 0.033,
    ) -> Tuple[float, float]:
        
        """
        Computes desired spherical tilt angles (alpha, beta) from ball and target positions.
        
        This is where the PID stuff happens
        TLDR: 
            error = target_pos - ball_pos
            desired tilt = kp * error + ki * integral(error) - kd * velocity
            return (desired tilt converted to spherical coords)
        """

        bx, by = ball_pos
        tx, ty = target_pos

        # Position error
        ex = tx - bx
        ey = ty - by

        # Velocity estimation (derivative on measurement to prevent derivative kick)
        if self.prev_ball_pos is None or dt <= 0:
            vx, vy = 0.0, 0.0
        else:
            vx = (bx - self.prev_ball_pos[0]) / dt
            vy = (by - self.prev_ball_pos[1]) / dt
        self.prev_ball_pos = (bx, by)

        # Integral accumulation with anti-windup clamping
        self.integral_x = max(-50.0, min(50.0, self.integral_x + ex * dt))
        self.integral_y = max(-50.0, min(50.0, self.integral_y + ey * dt))

        # Desired acceleration / tilt components
        ux = self.kp * ex + self.ki * self.integral_x - self.kd * vx
        uy = self.kp * ey + self.ki * self.integral_y - self.kd * vy

        # Convert cartesian tilt command to spherical angles (alpha, beta)
        mag = math.hypot(ux, uy)
        sin_alpha = min(mag, math.sin(self.max_tilt_rad))
        alpha = math.asin(sin_alpha)
        beta = math.atan2(uy, ux) if mag > 1e-9 else 0.0

        return alpha, beta




    def inverse_kinematics(
        self, alpha: float, beta: float
    ) -> Tuple[float, float, float]:
        
        """
        Compute the three servo angles needed to achieve the required tilt returned by compute_pid_tilt()
        """

        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        k = 1.0 - cos_a

        # 1. Induced yaw angle (psi) required to keep joints in their 120-deg vertical planes
        psi_num = -k * math.sin(2.0 * beta)
        psi_den = 2.0 * (1.0 + cos_a) + k * math.cos(2.0 * beta)
        psi = math.atan2(psi_num, psi_den)

        cos_p = math.cos(psi)
        sin_p = math.sin(psi)
        cos_b = math.cos(beta)
        sin_b = math.sin(beta)

        # 2. Rotated platform local axes u and v
        ux = (1.0 - k * cos_b**2) * cos_p - (k * cos_b * sin_b) * sin_p
        uy = -(k * cos_b * sin_b) * cos_p + (1.0 - k * sin_b**2) * sin_p
        uz = -sin_a * math.cos(beta - psi)

        vx = -(1.0 - k * cos_b**2) * sin_p - (k * cos_b * sin_b) * cos_p
        vy = (k * cos_b * sin_b) * sin_p + (1.0 - k * sin_b**2) * cos_p
        vz = -sin_a * math.sin(beta - psi)

        # 3. Parasitic horizontal shift of the platform center
        xc = -self.R * vx
        yc = -0.5 * self.R * (ux - vy)

        # 4. Ball joint coordinates A1, A2, A3 in base frame
        sqrt3_2 = math.sqrt(3.0) / 2.0

        y1 = yc + self.R * vy
        z1 = self.H + self.R * vz

        x2 = xc + self.R * (sqrt3_2 * ux - 0.5 * vx)
        z2 = self.H + self.R * (sqrt3_2 * uz - 0.5 * vz)

        x3 = xc + self.R * (-sqrt3_2 * ux - 0.5 * vx)
        z3 = self.H + self.R * (-sqrt3_2 * uz - 0.5 * vz)

        # 5. Radial reach (r_i) from each servo S_i to ball joint A_i
        r1 = y1 - self.L
        r2 = (x2 / sqrt3_2) - self.L
        r3 = (-x3 / sqrt3_2) - self.L

        # 6. Solve 2-link planar inverse kinematics for each servo
        theta1 = self._solve_2link_servo(r1, z1, arm_id=1)
        theta2 = self._solve_2link_servo(r2, z2, arm_id=2)
        theta3 = self._solve_2link_servo(r3, z3, arm_id=3)

        return math.degrees(theta1), math.degrees(theta2), math.degrees(theta3)


    

    def _solve_2link_servo(self, r: float, z: float, arm_id: int) -> float:

        """
        Solves for the angle of the driven link in plane 
        (basically a generic 2-link solver)
        (i have no idea how it works)
        """

        d_sq = r**2 + z**2
        d = math.sqrt(d_sq)

        if d > (self.l1 + self.l2) or d < abs(self.l1 - self.l2):
            raise ValueError(
                f"Target unreachable for Arm {arm_id}: distance d={d:.2f} mm "
                f"outside [{abs(self.l1 - self.l2):.2f}, {self.l1 + self.l2:.2f}] mm."
            )

        cos_phi = (d_sq + self.l1**2 - self.l2**2) / (2.0 * self.l1 * d)
        cos_phi = max(-1.0, min(1.0, cos_phi))

        return math.atan2(z, r) + self.elbow_sign * math.acos(cos_phi)




    def angles_to_pwm(
        self, angles_deg: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        
        """Maps raw geometric angles (deg) to calibrated PWM values for S1, S2, S3."""

        c1, c2, c3 = self.calibrations
        a1, a2, a3 = angles_deg
        return (
            c1.angle_to_pwm(a1),
            c2.angle_to_pwm(a2),
            c3.angle_to_pwm(a3),
    )

    def step(
        self,
        ball_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        dt: float = 0.016,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        
        """
        The BIG BIG FUNCTION that calls all the small functions:
            INPUTS: (ball_x, ball_y), (target_x, target_y) (in mm),
            OUTPUTS: ((theta_1, theta_2, theta_3), (pwm_1, pwm_2, pwm_3)).
        """

        alpha, beta = self.compute_pid_tilt(ball_pos, target_pos, dt)
        angles = self.inverse_kinematics(alpha, beta)
        pwms = self.angles_to_pwm(angles)
        return angles, pwms


def run_test_case(
    platform: StewartPlatformController,
    label: str,
    ball_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    dt: float = 0.0333,
    reset: bool = True,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    
    """
    Runs a test scenario on the platform controller and prints both
    the geometric angles and calibrated PWM values.

    Since the solver is meant to be called every frame, each test case represents 1 frame of output
    """

    if reset:
        platform.reset_controller()

    angles, pwms = platform.step(ball_pos, target_pos, dt=dt)
    print(
        f"{label:<24} | Ball: {str(ball_pos):<12} -> Target: {str(target_pos):<10} | "
        f"Angles: ({angles[0]:5.2f}°, {angles[1]:5.2f}°, {angles[2]:5.2f}°) | "
        f"PWM: ({pwms[0]:7.1f}, {pwms[1]:7.1f}, {pwms[2]:7.1f})"
    )
    return angles, pwms


# ==========================================
# Test Cases + Manual Calibration
# ==========================================
if __name__ == "__main__":
    # CLANKER COMMENTS:
    # Hardcoded two-point calibration per servo: (angle_deg, pwm_value)
    # Example values below (i replaced them) use pulse width in microseconds (us), e.g. at 0 deg and 60 deg.
    # You can also use PCA9685 ticks or duty cycle units as the lerp is identical.

    """
    ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS ALEX READ THIS 

    CALIBRATION COMMENTS:
    - 60 degrees is about 8%
    - OUTPUT ORDER IS (CHANNEL 2, CHANNEL 3, CHANNEL 0)
    - EXAMPLE: angles, pwms = platform.step(ball_pos, target_pos, dt=dt)
    - OUTPUTS:
        - angles = (CH2 angle, CH3 angle, CH0 angle)
        - pwms = (CH2 PWM %, CH3 PWM %, CH0 PWM %)
    """
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

    # init class
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

    # test cases

    run_test_case(
        platform,
        "Flat (Ball at center)",
        ball_pos=(0.0, 0.0),
        target_pos=(0.0, 0.0),
    )
    run_test_case(
        platform,
        "Ball near S2 (+X, -Y)",
        ball_pos=(30.0, -20.0),
        target_pos=(0.0, 0.0),
    )
    run_test_case(
        platform,
        "Ball near S1 (+Y)",
        ball_pos=(0.0, 40.0),
        target_pos=(0.0, 0.0),
    )
    run_test_case(
        platform,
        "Offset Target",
        ball_pos=(-15.0, -15.0),
        target_pos=(20.0, 10.0),
    )
    run_test_case(
        platform, 
        "Edge Case (literal)", 
        ball_pos=(99, -99), 
        target_pos=(0, 0)
    )
    run_test_case(
        platform, 
        "Edge Edge Case (x to -x)", 
        ball_pos=(140, 0), 
        target_pos=(-140, 0)
    )
    run_test_case(
        platform, 
        "Edge Edge Case (y to -y)", 
        ball_pos=(0, 140), 
        target_pos=(0, -140)
    )