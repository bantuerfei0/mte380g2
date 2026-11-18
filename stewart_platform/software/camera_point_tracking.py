import time
import cv2
import numpy as np

NAMES = ('top', 'center', 'right')
MAX_MISSED_FRAMES = 5  # Keep identities for this many invalid frames.
MAX_STEP_PX = 35
START_RADIUS_PX = 10


def measure(points):
    """Calculate image angle T-C-R and distances; coordinates are [x, y]."""
    top, center, right = (points[name] for name in NAMES)
    u, v = top-center, right-center
    tc, rc, tr = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(top-right)
    if min(tc, rc, tr) < 1:
        return None
    alpha = np.degrees(np.arccos(np.clip(np.dot(u, v)/(tc*rc), -1, 1)))
    return dict(alpha_deg=float(alpha), distance_tc_px=float(tc),
                distance_rc_px=float(rc), distance_tr_px=float(tr))


class CameraSensor:
    def __init__(self, device_index=0):
        self.hue, self.saturation, self.brightness = 6, 102, 148
        self.points, self.reference, self.stable = None, None, 0
        self.last_points, self.missed = None, 0
        self.candidates = []
        self.status = 'Searching: hold dots in starting orientation'
        self.settings = None
        self.capture = cv2.VideoCapture(device_index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError('Cannot open camera; try device_index=1')

    def reset(self):
        self.points, self.reference, self.stable = None, None, 0
        self.last_points, self.missed = None, 0
        self.status = 'Searching: return to starting orientation'

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = max(1, min(30, int(self.hue)))
        low = (0, int(self.saturation), int(self.brightness))
        high = (180-h, int(self.saturation), int(self.brightness))
        mask = cv2.inRange(hsv, low, (h, 255, 255))
        mask |= cv2.inRange(hsv, high, (179, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            if not 20 <= area <= 5000 or perimeter == 0:
                continue
            if 4*np.pi*area/perimeter**2 < .5:
                continue
            x, y, w, ht = cv2.boundingRect(contour)
            if x == 0 or y == 0 or x+w >= frame.shape[1] or y+ht >= frame.shape[0]:
                continue
            m = cv2.moments(contour)
            candidates.append(np.array([m['m10']/m['m00'], m['m01']/m['m00']]))
        return candidates, mask

    @staticmethod
    def label(dots):
        if len(dots) != 3:
            return None
        pairs = sorted((np.linalg.norm(dots[i]-dots[j]), i, j)
                       for i, j in ((0, 1), (0, 2), (1, 2)))
        if pairs[-1][0] < 1.1*pairs[-2][0]:
            return None
        _, i, j = pairs[-1]
        center = dots[3-i-j]
        for top, right in ((dots[i], dots[j]), (dots[j], dots[i])):
            u, v = top-center, right-center
            points = dict(top=top, center=center, right=right)
            g = measure(points)
            if (g and min(g['distance_tc_px'], g['distance_rc_px']) >= 15
                    and abs(np.degrees(np.arctan2(u[0], -u[1]))) <= 35
                    and abs(np.degrees(np.arctan2(-v[1], v[0]))) <= 35
                    and 60 <= g['alpha_deg'] <= 120):
                return points
        return None

    def track(self, dots):
        # last_points preserves identities; points only contains CURRENT valid data.
        self.points = None
        if self.last_points is not None:
            matches = [[i for i, p in enumerate(dots)
                        if np.linalg.norm(p-self.last_points[n]) <= MAX_STEP_PX]
                       for n in NAMES]
            if (all(len(m) == 1 for m in matches)
                    and len({m[0] for m in matches}) == 3):
                self.points = {n: dots[m[0]] for n, m in zip(NAMES, matches)}
                self.last_points, self.missed = self.points, 0
                self.status = 'Tracking'
                return
            self.missed += 1
            if self.missed <= MAX_MISSED_FRAMES:
                self.status = f'Temporary dot loss/ambiguity ({self.missed}/{MAX_MISSED_FRAMES}); no measurement'
                return
            self.reset()
            self.status = 'Tracking lost: return to starting orientation'
            return
        points = self.label(dots)
        if points is None:
            # Short detection flickers do not immediately erase startup progress.
            self.missed += 1
            if self.missed > MAX_MISSED_FRAMES:
                self.reference, self.stable = None, 0
            self.status = f'Searching: need 3 dots in starting layout (found {len(dots)})'
            return
        self.missed = 0
        if self.reference is None or any(np.linalg.norm(points[n]-self.reference[n]) > START_RADIUS_PX for n in NAMES):
            self.reference, self.stable = points, 1
        else:
            self.stable += 1
        self.status = f'Hold approximately still: {self.stable}/8'
        if self.stable >= 8:
            self.points, self.last_points, self.status = points, points, 'Tracking'

    def read(self):
        ok, frame = self.capture.read()
        acquired_at = time.perf_counter()
        if not ok:
            self.reset()
            raise RuntimeError('Camera stopped returning frames')
        settings = (self.hue, self.saturation, self.brightness)
        if settings != self.settings:
            self.reset()
            self.settings = settings
        self.candidates, mask = self.detect(frame)
        self.track(self.candidates)
        result = measure(self.points) if self.points is not None else None
        if result is not None:
            result['timestamp'] = acquired_at
        return frame, mask, self.points, result

    def close(self):
        self.capture.release()
