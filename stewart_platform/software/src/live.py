"""Live window and control-loop orchestration. Hardware calibration runs disarmed."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from datetime import datetime
import threading
import time

import cv2
import numpy as np

from .display import Window, label
from .calibration import configure
from .control import BallController
from .settings import RECORDINGS
from .vision import Camera, Tracker


class LiveBalancer:
    def __init__(self, config, path):
        self.config, self.path = config, path
        self.tracker = Tracker(config, path)
        self.controller = BallController()
        self.hardware = ExitStack()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.stop = threading.Event()
        self.future = self.drive = self.imu = None
        self.frame = self.ball = self.timestamp = self.hold_start = None
        self.goals = [np.zeros(2)]
        self.goal_index = 0
        self.action = self.click_mode = ""
        self.mode = "preview"
        self.armed = False
        self.midpoint = self.mask_view = False
        self.message = "Hold the plate physically level before capturing its reference."
        self.status = "Waiting for camera"
        self.log = self.writer = None
        self.history = deque()
        self.view = np.zeros((360,640,3),np.uint8)

    def run(self, initial_mode=None):
        setup_actions = {"Capture level":"level", "Read tilt":"read", "Probe (ball off)":"identify",
                         "Use home":"home", "Use continuous":"continuous"}
        click_actions = {"Mark 3 dots":"dots", "Sample ball":"ball", "Set goal":"goal", "Append goals":"append"}
        actions = {"Save vision":self.save_vision, "Centre":self.centre, "Arm / disarm":self.toggle_arm,
                   "Record / stop":self.record}
        window = Window("Live ball balancer")
        self.camera = None
        try:
            self.camera = Camera(self.config["camera_index"])
            for gain, maximum in (("kp",100),("ki",20),("kd",100)):
                cv2.createTrackbar(gain+" x100",window.title,round(100*getattr(self.controller,gain)),maximum,
                                   lambda value,k=gain: setattr(self.controller,k,value/100))
            if initial_mode in ("home","continuous"):
                self.setup(initial_mode)
            while True:
                self.tick()
                midpoint_button = f"Midpoint: {'on' if self.midpoint else 'off'}"
                mask_button = f"Mask: {'on' if self.mask_view else 'off'}"
                buttons = [*setup_actions,*click_actions,*actions,midpoint_button,mask_button,"Back"]
                button,click,key = window.poll(self.view,buttons,self.status+"\n"+self.message)
                if key in (27,ord("q")) or button == "Back":
                    break
                if key in (10,13):
                    self.click_mode = ""
                if button in setup_actions:
                    self.setup(setup_actions[button])
                elif button in click_actions:
                    self.select_click(click_actions[button])
                elif button in actions:
                    actions[button]()
                elif button == midpoint_button:
                    self.midpoint = not self.midpoint
                elif button == mask_button:
                    self.mask_view = not self.mask_view
                elif click and self.frame is not None and click[1] < self.frame.shape[0]:
                    self.click(*click)
        finally:
            try:
                self.close()
            finally:
                window.close()

    def idle(self):
        self.armed = False
        self.hold_start = None
        self.controller.reset()
        if self.drive:
            self.drive.idle()

    def setup(self, action):
        if self.future:
            return
        self.idle()
        self.hardware.close()
        self.drive = self.imu = None
        self.mode, self.action = "preview", action
        self.message = "Measuring servo responses..." if action == "identify" else "Setting up..."
        self.future = self.worker.submit(configure, action, self.config, self.path, self.hardware, self.stop)

    def toggle_arm(self):
        if self.armed:
            self.idle()
        elif self.future is None and self.ball is not None and time.monotonic()-self.timestamp < .2:
            self.armed = True
            self.message = "Tracking in preview; PWM off." if self.drive is None else "Balancing."

    def centre(self):
        self.goals, self.goal_index, self.hold_start = [np.zeros(2)], 0, None
        self.click_mode = ""

    def select_click(self, mode):
        if mode in ("dots","ball"):
            self.idle()
        if mode == "dots":
            self.tracker.templates.clear()
            self.tracker.transform = None
        self.click_mode = mode
        self.message = {"dots":"Click dot 1 in measured order.", "ball":"Click inside the ball.",
                        "goal":"Click a goal on the plate.", "append":"Click waypoints. Enter finishes."}[mode]

    def click(self, x, y):
        if self.frame is None:
            return
        try:
            if self.click_mode == "dots":
                self.tracker.add_dot(self.frame, x, y)
                count = len(self.tracker.templates)
                self.message = f"Click dot {count+1}." if count < 3 else "Three dots selected."
                if count == 3:
                    self.click_mode = ""
            elif self.click_mode == "ball":
                self.tracker.sample_ball(self.frame, x, y)
                self.click_mode = ""
                self.message = "Ball colour sampled."
            elif self.click_mode in ("goal","append") and self.tracker.transform is not None:
                point = self.tracker.transform @ [x,y,1]
                if np.linalg.norm(point) > self.tracker.radius:
                    raise ValueError("Choose a goal inside the working radius.")
                if self.click_mode == "goal":
                    self.goals = [point]
                    self.click_mode = ""
                else:
                    self.goals.append(point)
                self.goal_index, self.hold_start = 0, None
        except ValueError as exc:
            self.message = str(exc)

    def save_vision(self):
        try:
            self.tracker.save()
            self.message = "Vision calibration saved."
        except (ValueError, OSError) as exc:
            self.message = str(exc)

    def record(self):
        if self.log:
            self.log.close()
            self.log = self.writer = None
            self.message = "Recording saved."
            return
        RECORDINGS.mkdir(exist_ok=True)
        path = RECORDINGS / (datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".csv")
        self.log = path.open("x", newline="")
        self.writer = csv.writer(self.log)
        self.writer.writerow(("t_s mode armed midpoint kp ki kd valid goal_x_cm goal_y_cm aim_x_cm aim_y_cm "
                              "ball_x_cm ball_y_cm target_x_deg target_y_deg imu_x_deg imu_y_deg pwm1_us pwm2_us pwm3_us").split())
        self.started = time.monotonic()
        self.message = f"Recording {path.name}"

    def tick(self):
        try:
            if self.future and self.future.done():
                future, self.future = self.future, None
                self.imu, self.drive, message = future.result()
                self.mode = self.action if self.drive else "preview"
                self.message = message
            latest = self.camera.latest
            now = time.monotonic()
            if latest is None or latest[1] is None or now-latest[0] > .2:
                if self.armed:
                    self.idle()
                if self.writer and self.timestamp is not None:
                    missing = [np.nan, np.nan]
                    pid = self.controller
                    pulses = self.drive.servos.pulses if self.drive else [np.nan]*3
                    self.writer.writerow([now-self.started, self.mode, 0, int(self.midpoint),
                                          pid.kp, pid.ki, pid.kd, 0, *self.goals[self.goal_index],
                                          *missing, *missing, 0., 0., *missing, *pulses])
                self.ball = self.frame = self.timestamp = self.tracker.transform = None
                self.view.fill(0)
                label(self.view,"Waiting for camera",(12,28))
                self.status = f"{self.mode} | idle"
            elif latest[0] != self.timestamp:
                self.update(*latest)
        except Exception as exc:
            self.armed = False
            self.drive = self.imu = None
            self.hardware.close()
            self.controller.reset()
            self.mode = "preview"
            self.message = str(exc)

    def update(self, stamp, frame):
        dt = stamp-self.timestamp if self.timestamp is not None else 0.
        self.timestamp, self.frame = stamp, frame
        self.ball, mask, dots = self.tracker.detect(frame)
        if self.armed and (self.ball is None or dt > .2):
            self.idle()
        ball = self.ball if self.ball is not None else np.full(2,np.nan)
        goal = self.goals[self.goal_index]
        if self.armed and np.linalg.norm(ball-goal) <= .5:
            if self.hold_start is None:
                self.hold_start = stamp
            elif stamp-self.hold_start >= 1. and len(self.goals) > 1:
                self.goal_index = (self.goal_index+1) % len(self.goals)
                goal = self.goals[self.goal_index]
                self.hold_start = None
        else:
            self.hold_start = None
        measured = self.imu.tilt() if self.imu else np.full(2,np.nan)
        midpoint = self.midpoint
        aim = (ball+goal)/2 if midpoint else goal
        command = np.zeros(2)
        if self.armed:
            aim, command = self.controller.update(ball, goal, dt, midpoint, self.config["max_tilt_deg"])
            if self.drive:
                self.drive.update(command, measured, dt)
        if self.writer:
            pid = self.controller
            pulses = self.drive.servos.pulses if self.drive else [np.nan]*3
            self.writer.writerow([stamp-self.started,self.mode,int(self.armed),int(midpoint),pid.kp,pid.ki,pid.kd,
                                  int(self.ball is not None),*goal,*aim,*ball,*command,*measured,*pulses])
        image = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) if self.mask_view else frame.copy()
        for index, dot in enumerate(dots):
            cv2.putText(image,str(index+1),dot,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        if self.tracker.transform is not None:
            inverse = cv2.invertAffineTransform(self.tracker.transform)
            for point, ink in [(p,(180,100,255)) for p in self.goals]+[(aim,(0,210,255)),(goal,(0,0,255)),(ball,(0,255,0))]:
                if np.isfinite(point).all():
                    cv2.circle(image,tuple(np.rint(inverse @ [*point,1]).astype(int)),6,ink,2)
        self.status = (f"{self.mode} | {'armed' if self.armed else 'idle'} | dots {len(dots)}/3\n"
                       f"Goal {self.goal_index+1}/{len(self.goals)}: {goal.round(1)} cm\n"
                       f"Ball {ball.round(1)} cm\nTilt {measured.round(1)} deg")
        self.history.append((stamp,ball,goal))
        while self.history and stamp-self.history[0][0] > 10:
            self.history.popleft()
        width = image.shape[1]
        chart = np.full((120,width,3),32,np.uint8)
        for axis in range(2):
            baseline = 30+60*axis
            cv2.line(chart,(0,baseline),(width-1,baseline),(85,85,85))
            for column,ink in [(2,(60,60,255)),(1,(90,220,100))]:
                line = []
                for sample in self.history:
                    value = sample[column][axis]
                    if np.isfinite(value):
                        line.append((round((width-1)*(1+(sample[0]-stamp)/10)),
                                     round(baseline-26*value/self.tracker.radius)))
                    else:
                        if len(line)>1:
                            cv2.polylines(chart,[np.int32(line)],False,ink,1)
                        line.clear()
                if len(line)>1:
                    cv2.polylines(chart,[np.int32(line)],False,ink,1)
            label(chart,f"{'XY'[axis]} +/-{self.tracker.radius:g} cm",(5,baseline-12))
        self.view = np.vstack((image,chart))

    def close(self):
        self.stop.set()
        self.worker.shutdown(wait=True, cancel_futures=True)
        try:
            self.hardware.close()
        finally:
            if self.log:
                self.log.close()
            if self.camera:
                self.camera.close()
