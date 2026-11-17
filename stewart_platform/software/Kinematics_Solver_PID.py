import math
from typing import Tuple, Optional

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

class StewartPlatformController:
    def __init__(
        self,
        R: float,   # radius from centre of platform to ball joint (top of linkage)
        L: float,   # length from centre of base to servo actuation point (bottom of linkage)
        H: float,   # z-offset of the platform. Changes the idle servo angle (angle when ball_pos = target_pos)
        l1: float,  # the length of the driven link (i.e. the one connected to the servo)
        l2: float,  # the length of the second link (i.e. the one connected to the ball joint)
        max_tilt_deg: float = 15.0,     # if the tilt is too much parallax will fuck up the camera but i dont wanna think about that
        kp: float = 0.005,
        kd: float = 0.003,
        ki: float = 0.0,
        elbow_up: bool = False,     # False for linkage "elbows" flaring outwards (ideally never True)
    ):

        # even i know what this does
        self.R = R
        self.L = L
        self.H = H
        self.l1 = l1
        self.l2 = l2
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
        dt: float = 0.016,
    ) -> Tuple[float, float]:
        
        """
        Computes desired spherical tilt angles (alpha, beta) given ball and target positions
        
        The spherical tilt angles are the spherical coordinates of the normal vector of the platform plane. 
        
        alpha is the tilt away from the z-axis, beta is the tilt from the x-axis (CCW in the x-y plane)
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
        # Note: Negative signs tilt the high side toward the ball so gravity rolls it toward target
        mag = math.hypot(ux, uy)
        sin_alpha = min(mag, math.sin(self.max_tilt_rad))
        alpha = math.asin(sin_alpha)
        # Change this line in compute_pid_tilt:
        beta = math.atan2(uy, ux) if mag > 1e-9 else 0.0

        return alpha, beta

    def inverse_kinematics(self, alpha: float, beta: float) -> Tuple[float, float, float]:

        """
        Computes the three servo angles (in degrees) for spherical tilt angles (alpha, beta),
        accounting for planar linkage constraints and parasitic center shift.
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
        inv_sqrt3 = 1.0 / math.sqrt(3.0)

        # A1 (along +Y plane)
        y1 = yc + self.R * vy
        z1 = self.H + self.R * vz

        # A2 (+120 deg plane (aka 120 deg CW) (positive x side))
        x2 = xc + self.R * (sqrt3_2 * ux - 0.5 * vx)
        z2 = self.H + self.R * (sqrt3_2 * uz - 0.5 * vz)

        # A3 (-120 deg plane (aka 120 deg CCW) (negative x side))
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
        Solves the 2-link triangle in the vertical plane of a single arm.
        """

        d_sq = r**2 + z**2
        d = math.sqrt(d_sq)

        # Workspace reachability check (basically stop if its trying to go somewhere longer than l1 + l2)
        if d > (self.l1 + self.l2) or d < abs(self.l1 - self.l2):
            raise ValueError(
                f"Target unreachable for Arm {arm_id}: distance d={d:.2f} mm "
                f"outside [{abs(self.l1 - self.l2):.2f}, {self.l1 + self.l2:.2f}] mm."
            )

        cos_phi = (d_sq + self.l1**2 - self.l2**2) / (2.0 * self.l1 * d)
        cos_phi = max(-1.0, min(1.0, cos_phi))  # Clamp against floating-point drift

        theta = math.atan2(z, r) + self.elbow_sign * math.acos(cos_phi)
        return theta

    def step(
        self,
        ball_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        dt: float = 0.016,
    ) -> Tuple[float, float, float]:
        """
        Main update function: takes (ball_x, ball_y) and (target_x, target_y) in mm,
        and returns (theta_1, theta_2, theta_3) in degrees.
        """
        alpha, beta = self.compute_pid_tilt(ball_pos, target_pos, dt)
        return self.inverse_kinematics(alpha, beta)



def run_test_case(
    platform: StewartPlatformController,
    label: str,
    ball_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    dt: float = 0.016,
    reset: bool = True,
) -> Tuple[float, float, float]:
    
    """
    Runs a single test scenario on the platform controller and prints the resulting servo angles.
    """

    if reset:
        platform.reset_controller()
        
    angles = platform.step(ball_pos, target_pos, dt=dt)
    print(
        f"{label:<28} | Ball: {str(ball_pos):<12} -> Target: {str(target_pos):<10} | "
        f"S1={angles[0]:6.2f}°, S2={angles[1]:6.2f}°, S3={angles[2]:6.2f}°"
    )
    return angles

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Stewart Platform dimensions (in mm)
    platform = StewartPlatformController(
        R = 140,                # Platform radius (distance from platform centre to ball joints)
        L = 54,                 # Base servo radius (xy-radius from servos to base centre)
        H = 100,                # "Default" height (z-offset from servo acutation point to table)
        l1 = 80,                # Driven link length
        l2 = 94,                # Ball joint link length
        max_tilt_deg=10,        # maximum table tilt
        kp=0.002,
        kd=0.001,
        ki=0.0,
        elbow_up=False,         # Elbows point outward
    )

    print("Kp: " + str(platform.kp) + ", Kd: " + str(platform.kd) + ", Ki: " + str(platform.ki))
    run_test_case(platform, "Flat (idle case)", ball_pos=(0.0, 0.0), target_pos=(0.0, 0.0))
    run_test_case(platform, "Ball near S2 (+X, -Y)", ball_pos=(30.0, -20.0), target_pos=(0.0, 0.0))
    run_test_case(platform, "Ball near S1 (+Y)", ball_pos=(0.0, 40.0), target_pos=(0.0, 0.0))
    run_test_case(platform, "Offset Target", ball_pos=(-15.0, -15.0), target_pos=(20.0, 10.0))
    run_test_case(platform, "Edge Case (literal)", ball_pos=(99, -99), target_pos=(0, 0))