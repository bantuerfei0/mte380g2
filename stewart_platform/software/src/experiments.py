"""Simulated releases and analysis of the same measurements in hardware CSVs."""
import csv
from pathlib import Path

import numpy as np

from .calibration import home
from .control import BallController, TiltDrive, inverse_response


class ModelPlate:
    def __init__(self, bias):
        self.matrix = np.array([[0, .030, -.024], [.031, -.012, -.014]])
        self.bias = np.array(bias, float)
        self.park = np.array([1500., 1520., 1480.])
        self.low, self.high = self.park-180, self.park+180
        self.pulses = self.park.copy()

    def write(self, pulses):
        self.pulses = np.clip(pulses, self.low, self.high)

    def tilt(self):
        return self.matrix @ (self.pulses-self.park) + self.bias


def simulate(kd, midpoint=False, disturbance=False, continuous=False):
    plate = ModelPlate([.8, -.5] if disturbance else [0, 0])
    inverse = inverse_response(plate.matrix)
    rest = plate.park if continuous else home(plate, inverse, plate.tilt)
    drive = TiltDrive(plate, inverse, rest, continuous)
    controller = BallController()
    controller.kd = kd
    tilt = plate.tilt()
    position, velocity, goal = np.array([-6.,0.]), np.zeros(2), np.zeros(2)
    dt = .02
    rows = []
    for step in range(501):
        if disturbance and step == 200:
            plate.bias += [1., -.6]
        _, target = controller.update(position, goal, dt, midpoint, 4)
        drive.update(target, tilt, dt)
        rows.append([step*dt, *position, *goal])
        tilt += (1-np.exp(-dt/.07))*(plate.tilt()-tilt)
        acceleration = .6*981*np.sin(np.radians(tilt)) - .25*velocity
        position += velocity*dt + .5*acceleration*dt**2
        velocity += acceleration*dt
    return np.array(rows)


def recording(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    active = [i for i,r in enumerate(rows) if r["armed"] == "1"]
    if len(active) < 2 or active != list(range(active[0],active[-1]+1)):
        raise ValueError("Record one uninterrupted armed trial.")
    rows = rows[active[0]:active[-1]+1]
    settings = "mode midpoint goal_x_cm goal_y_cm kp ki kd".split()
    if any(r["valid"] != "1" or any(r[k] != rows[0][k] for k in settings) for r in rows):
        raise ValueError("Keep goal, controller settings and tracking fixed during a trial.")
    data = np.array([[float(r[k]) for k in ("t_s","ball_x_cm","ball_y_cm","goal_x_cm","goal_y_cm")] for r in rows])
    data[:,0] -= data[0,0]
    return data


def report(paths=(), show=False, plot=None):
    if paths:
        trials = [(Path(path).stem, recording(path)) for path in paths]
    else:
        trials = [("Light damping", simulate(.12)), ("Damped", simulate(.38)),
                  ("Damped + midpoint", simulate(.38, midpoint=True)),
                  ("Startup home", simulate(.38, disturbance=True)),
                  ("Continuous IMU", simulate(.38, disturbance=True, continuous=True))]
    summary = []
    for name, data in trials:
        error = data[:,1:3]-data[:,3:5]
        length = np.linalg.norm(error[0])
        if length < 1e-6:
            raise ValueError("Start away from the goal.")
        crossing = max(0., np.max(error @ (-error[0]/length)))
        distance = np.linalg.norm(error, axis=1)
        outside = np.flatnonzero(distance > .5)
        first = outside[-1]+1 if outside.size else 0
        settled = f"{data[first,0]:.2f} s" if first < len(data) and data[-1,0]-data[first,0] >= 1 else "unsettled"
        summary.append(f"{name}: overshoot {crossing:.3f} cm, {settled}, final error {distance[-1]:.3f} cm")
    if show or plot:
        from .plots import response_plot
        response_plot(trials, bool(paths), show, plot)
    return summary
