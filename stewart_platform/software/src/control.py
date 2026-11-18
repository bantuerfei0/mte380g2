"""Outer ball position loop and inner plate tilt control."""
import numpy as np


class BallController:
    def __init__(self):
        self.kp, self.ki, self.kd = .35, 0., .38
        self.reset()

    def reset(self):
        self.previous = None
        self.velocity = np.zeros(2)
        self.integral = np.zeros(2)

    def update(self, position, goal, dt, midpoint, max_tilt):
        aim = (position + goal)/2 if midpoint else goal
        error = aim - position
        if self.previous is not None and dt > 0:
            self.velocity += .45*((position-self.previous)/dt - self.velocity)
            self.integral = np.clip(self.integral + dt*error, -3, 3)
        self.previous = position.copy()
        command = self.kp*error + self.ki*self.integral - self.kd*self.velocity
        return aim, command * min(1., max_tilt/max(1e-9, np.linalg.norm(command)))


def inverse_response(matrix):
    matrix = np.asarray(matrix, float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid servo response matrix.")
    largest, smallest = np.linalg.svd(matrix, compute_uv=False)
    if smallest < .001 or largest/smallest > 20:
        raise ValueError("Servo responses do not span both tilt directions.")
    return np.linalg.pinv(matrix)


class TiltDrive:
    def __init__(self, servos, inverse, rest, continuous):
        self.servos = servos
        self.inverse = inverse
        self.rest = rest.copy()
        self.continuous = continuous
        self.idle()

    def idle(self):
        self.trim = self.rest.copy()
        self.servos.write(self.rest)

    def update(self, target, measured, dt):
        feedforward = self.inverse @ target
        if self.continuous:
            correction = self.inverse @ (target - measured)
            self.trim = np.clip(self.trim + 2*dt*correction,
                                self.servos.low-feedforward, self.servos.high-feedforward)
        self.servos.write(self.trim + feedforward)
