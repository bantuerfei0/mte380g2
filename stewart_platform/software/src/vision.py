"""Latest camera frame, three-dot affine calibration and metric ball detection."""
import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index):
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Could not open camera {index}.")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                self.latest = (time.monotonic(), frame if ok else None)
                if not ok:
                    self.stop.wait(.03)
        finally:
            self.capture.release()

    def close(self):
        self.stop.set()
        self.thread.join(.5)


class Tracker:
    def __init__(self, config, path):
        self.path = path.with_suffix(".vision.npz")
        self.points = np.float32(config["marker_xy_cm"])
        self.index = config["camera_index"]
        self.radius = config["working_radius_cm"]
        self.templates = []
        self.color = self.transform = None
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                if np.array_equal(data["platform_points"], self.points) and data["camera_index"] == self.index:
                    self.templates = list(zip(data["anchors"], data["patches"]))
                    self.color = data["color"]

    def add_dot(self, frame, x, y):
        patch = frame[y-8:y+9, x-8:x+9]
        if patch.shape != (17, 17, 3) or patch.std(axis=(0, 1)).max() < 8:
            raise ValueError("Choose a high-contrast dot away from the image edge.")
        if any(np.linalg.norm(np.array([x,y])-anchor) < 50 for anchor, _ in self.templates):
            raise ValueError("Choose the next dot, at least 50 pixels from the others.")
        self.templates.append((np.array([x,y]), patch.copy()))

    def sample_ball(self, frame, x, y):
        patch = frame[y-2:y+3, x-2:x+3]
        if patch.shape != (5,5,3):
            raise ValueError("Click inside the ball.")
        self.color = cv2.cvtColor(np.uint8([[np.median(patch, axis=(0,1))]]), cv2.COLOR_BGR2HSV)[0,0].astype(int)

    def save(self):
        if self.transform is None or self.color is None or len(self.templates) != 3:
            raise ValueError("Track three dots and sample the ball first.")
        anchors, patches = zip(*self.templates)
        temporary = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, anchors=anchors, patches=patches, color=self.color,
                            platform_points=self.points, camera_index=self.index)
        temporary.replace(self.path)

    def detect(self, frame):
        self.transform = None
        dots = []
        for (x, y), patch in self.templates:
            left, top = max(0, x-30), max(0, y-30)
            region = frame[top:y+31, left:x+31]
            if min(region.shape[:2]) < 17:
                break
            result = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, (dx, dy) = cv2.minMaxLoc(result)
            if score < .78:
                break
            dots.append((left+dx+8, top+dy+8))
        mask = np.zeros(frame.shape[:2], np.uint8)
        if len(dots) != 3:
            return None, mask, dots
        pixels = np.float32(dots)
        triangle = pixels[1:] - pixels[0]
        if abs(np.linalg.det(triangle)) < 400 or np.linalg.cond(triangle) > 12:
            return None, mask, dots
        self.transform = cv2.getAffineTransform(pixels, self.points)
        if self.color is None:
            return None, mask, dots
        diff = abs(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(int) - self.color)
        hue_ok = (self.color[1] < 45) | (np.minimum(diff[:,:,0], 180-diff[:,:,0]) <= 12)
        mask = np.uint8(hue_ok & (diff[:,:,1] <= 65) & (diff[:,:,2] <= 70))*255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        area_scale = abs(np.linalg.det(self.transform[:,:2]))
        for contour in contours:
            moments = cv2.moments(contour)
            area = moments["m00"]
            if not 4 < area*area_scale < 23 or 4*np.pi*area/cv2.arcLength(contour, True)**2 < .65:
                continue
            point = self.transform @ [moments["m10"]/area, moments["m01"]/area, 1]
            if np.linalg.norm(point) <= self.radius:
                candidates.append(point)
        return candidates[0] if len(candidates) == 1 else None, mask, dots
