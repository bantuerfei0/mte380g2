"""
CLANK CLANK CLANK

1D ball-on-beam position detector with dynamic beam-end tracking.

Setup:
  * Stick a small colored marker (tape/sticker) on each end of the beam.
    Both markers share one color; it must differ from the ball's color.
  * Run, press `t` to tune HSV for the ball (`b`) and the markers (`m`).

Each frame:
  1. Find the two marker blobs -> beam ends (left = smaller x), lightly smoothed.
  2. Find the ball inside a band around the current beam line.
  3. Project the ball onto the beam -> pos in [-1, +1] (0 = centre).
  4. Kalman filter (constant-velocity) -> smoothed pos + velocity,
     and it keeps predicting through brief dropouts.

Keys:  t = toggle tuning   b / m = tune ball / markers   s = save   q = quit
Usage: python ball_position_tracked.py [camera_index]
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

CALIB_FILE = Path("calib_tracked.json")
WIN = "ball balancer"

BEAM_ALPHA = 0.6        # smoothing of beam-end positions (1 = no smoothing)
BAND_HALF_WIDTH = 40    # px, ball search band around the beam
MAX_LOST_FRAMES = 10    # reset the filter after this many missed frames
KF_ACCEL_NOISE = 5.0    # process noise (units/s^2); raise if the filter lags
KF_MEAS_NOISE = 1e-4    # measurement variance in normalized pos units

# defaults are in hsv
DEFAULTS = {
    "ball_low": [5, 120, 120],       # orange-ish ball
    "ball_high": [20, 255, 255],
    "marker_low": [100, 120, 80],    # blue markers
    "marker_high": [130, 255, 255],
    "ball_min_area": 80,
    "marker_min_area": 60,
}


# ---------------------------------------------------------------- settings
def load_settings():
    s = dict(DEFAULTS)
    if CALIB_FILE.exists():
        s.update(json.loads(CALIB_FILE.read_text()))
    return s


def save_settings(s):
    CALIB_FILE.write_text(json.dumps(s, indent=2))
    print(f"saved {CALIB_FILE}")


# ---------------------------------------------------------------- capture
class LatestFrame:
    """Grabs frames in a thread so we always process the NEWEST frame
    (avoids stale buffered frames adding latency)."""

    def __init__(self, cap):
        self.cap = cap
        self.frame, self.stamp, self.new = None, 0.0, False
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                t = time.perf_counter()
                with self.lock:
                    self.frame, self.stamp, self.new = f, t, True

    def read(self):
        while self.running:
            with self.lock:
                if self.new:
                    self.new = False
                    return self.frame, self.stamp
            time.sleep(0.001)
        return None, 0.0

    def stop(self):
        self.running = False


# ---------------------------------------------------------------- vision
def hsv_mask(hsv, low, high):
    m = cv2.inRange(hsv, np.array(low), np.array(high))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, None, iterations=2)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, None, iterations=2)


def blob_centers(mask, min_area, n=None):
    """Centroids (and equivalent radii) of the n largest blobs."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    cnts.sort(key=cv2.contourArea, reverse=True)
    out = []
    for c in cnts[:n]:
        m = cv2.moments(c)
        if m["m00"] > 0:
            r = np.sqrt(m["m00"] / np.pi)
            out.append((np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), r))
    return out


def update_beam(beam, markers):
    """beam: 2x2 array [[lx,ly],[rx,ry]] or None. Needs both markers to update."""
    if len(markers) < 2:
        return beam
    l, r = sorted([markers[0][0], markers[1][0]], key=lambda p: p[0])
    new = np.array([l, r])
    return new if beam is None else BEAM_ALPHA * new + (1 - BEAM_ALPHA) * beam


def project_on_beam(pt, beam):
    a, b = beam
    ab = b - a
    return float(np.dot(pt - a, ab) / np.dot(ab, ab))


def beam_band(shape, beam):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.line(mask, tuple(beam[0].astype(int)), tuple(beam[1].astype(int)),
             255, 2 * BAND_HALF_WIDTH)
    return mask


# ---------------------------------------------------------------- filter
class PosFilter:
    """Constant-velocity Kalman filter on normalized beam position.
    dt is measured every frame, so 30 fps jitter is handled properly."""

    def __init__(self):
        kf = cv2.KalmanFilter(2, 1)
        kf.measurementMatrix = np.array([[1, 0]], np.float32)
        kf.measurementNoiseCov = np.array([[KF_MEAS_NOISE]], np.float32)
        self.kf, self.ready = kf, False

    def reset(self):
        self.ready = False

    def step(self, dt, meas):
        kf = self.kf
        kf.transitionMatrix = np.array([[1, dt], [0, 1]], np.float32)
        g = np.array([[dt * dt / 2], [dt]], np.float32)
        kf.processNoiseCov = KF_ACCEL_NOISE * (g @ g.T) + 1e-6 * np.eye(2, dtype=np.float32)
        if not self.ready:
            if meas is None:
                return None
            kf.statePost = np.array([[meas], [0]], np.float32)
            kf.errorCovPost = np.eye(2, dtype=np.float32)
            self.ready = True
            return meas, 0.0
        est = kf.predict()
        if meas is not None:
            est = kf.correct(np.array([[meas]], np.float32))
        return float(est[0, 0]), float(est[1, 0])


# ---------------------------------------------------------------- tuning UI
def make_sliders(s, target):
    cv2.namedWindow("tuning")
    for n, m in zip(["H low", "S low", "V low", "H high", "S high", "V high"],
                    [179, 255, 255, 179, 255, 255]):
        cv2.createTrackbar(n, "tuning", 0, m, lambda _: None)
    load_sliders(s, target)


def load_sliders(s, target):
    vals = s[f"{target}_low"] + s[f"{target}_high"]
    for n, v in zip(["H low", "S low", "V low", "H high", "S high", "V high"], vals):
        cv2.setTrackbarPos(n, "tuning", int(v))


def read_sliders(s, target):
    g = lambda n: cv2.getTrackbarPos(n, "tuning")
    s[f"{target}_low"] = [g("H low"), g("S low"), g("V low")]
    s[f"{target}_high"] = [g("H high"), g("S high"), g("V high")]


# ---------------------------------------------------------------- main
def main():
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        sys.exit(f"could not open camera {cam_idx}")
    # MJPG usually allows a full 30 fps at 640x480; raw YUYV can be slower.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    grabber = LatestFrame(cap)

    s = load_settings()
    flt = PosFilter()
    beam, lost = None, 0
    tuning, target = False, "ball"
    prev_stamp, fps = None, 30.0

    while True:
        frame, stamp = grabber.read()
        if frame is None:
            break
        dt = 1 / 30 if prev_stamp is None else max(stamp - prev_stamp, 1e-3)
        prev_stamp = stamp
        fps = 0.9 * fps + 0.1 / dt

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
        vis = frame.copy()

        if tuning:
            read_sliders(s, target)

        # 1. beam ends from markers (search whole frame: the beam moves)
        m_mask = hsv_mask(hsv, s["marker_low"], s["marker_high"])
        markers = blob_centers(m_mask, s["marker_min_area"], n=2)
        beam = update_beam(beam, markers)
        for c, r in markers:
            cv2.circle(vis, tuple(c.astype(int)), int(r) + 3, (255, 255, 0), 2)

        result = None
        if beam is None:
            cv2.putText(vis, "beam markers not found", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.line(vis, tuple(beam[0].astype(int)), tuple(beam[1].astype(int)),
                     (255, 0, 0), 2)

            # 2. ball inside a band around the beam
            b_mask = cv2.bitwise_and(hsv_mask(hsv, s["ball_low"], s["ball_high"]),
                                     beam_band(frame.shape, beam))
            balls = blob_centers(b_mask, s["ball_min_area"], n=1)

            # 3. project onto beam -> normalized position
            meas = None
            if balls:
                c, r = balls[0]
                meas = 2 * project_on_beam(c, beam) - 1
                cv2.circle(vis, tuple(c.astype(int)), int(r), (0, 255, 0), 2)

            # 4. filter (predicts through short dropouts)
            lost = 0 if meas is not None else lost + 1
            if lost > MAX_LOST_FRAMES:
                flt.reset()
            result = flt.step(dt, meas)

        if result is not None:
            pos, vel = result
            cv2.putText(vis, f"pos {pos:+.3f}  vel {vel:+.2f}/s  {fps:.0f} fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # ---> feed `pos` and `vel` into your PID / controller here <---
        elif beam is not None:
            cv2.putText(vis, "ball not found", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(WIN, vis)
        if tuning:
            cv2.imshow("mask", hsv_mask(hsv, s[f"{target}_low"], s[f"{target}_high"]))
            cv2.setWindowTitle("tuning", f"tuning: {target}")

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            save_settings(s)
        elif k == ord("t"):
            tuning = not tuning
            if tuning:
                make_sliders(s, target)
            else:
                cv2.destroyWindow("tuning")
                cv2.destroyWindow("mask")
        elif k in (ord("b"), ord("m")) and tuning:
            target = "ball" if k == ord("b") else "marker"
            load_sliders(s, target)

    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()