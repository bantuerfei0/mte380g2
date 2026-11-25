import time
import cv2
import numpy as np

NAMES = ('top', 'center', 'right')
MAX_MISSED_FRAMES = 5  # Keep identities for this many invalid frames.
MAX_STEP_PX = 35
START_RADIUS_PX = 10
ROTATION_TOLERANCE_DEG = 1.0
RIGHT_ANGLE_TOLERANCE_DEG = 2.0
CENTER_TOLERANCE_PX = 10.0
READY_HOLD_SECONDS = 1.0


def measure(points):
    """Calculate image angle T-C-R and distances; coordinates are [x, y]."""
    top, center, right = (points[name] for name in NAMES)
    u, v = top-center, right-center
    tc, rc, tr = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(top-right)
    if min(tc, rc, tr) < 1:
        return None
    alpha = np.degrees(np.arccos(np.clip(np.dot(u, v)/(tc*rc), -1, 1)))
    return dict(alpha_deg=float(np.degrees(np.arctan2(u[0], -u[1]))),
                internal_angle_deg=float(alpha), distance_tc_px=float(tc),
                distance_rc_px=float(rc), distance_tr_px=float(tr))


class CameraSensor:
    def __init__(self, device_index=0):
        self.hue, self.saturation, self.brightness = 19, 133, 54
        self.points, self.reference, self.stable = None, None, 0
        self.last_points, self.missed = None, 0
        self.candidates = []
        self.status = 'Searching: hold dots in starting orientation'
        self.settings = None
        self._distance_cache = {}  # name -> (pixels, last update timestamp)
        self._fresh_distances = set()
        self.ready, self.aligned_since = False, None
        self.calibration_status = 'Aligning'
        self.capture = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError('Cannot open camera; try device_index=1')

    def reset(self):
        self._fresh_distances.clear()  # Retain distances through tracking resets.
        self.points, self.reference, self.stable = None, None, 0
        self.last_points, self.missed = None, 0
        self.status = 'Searching: return to starting orientation'
        self.ready, self.aligned_since = False, None
        self.calibration_status = 'Aligning'

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = max(1, min(30, int(self.hue)))
        blue_hue = 110

        lower = (
            max(0, blue_hue - h),
            int(self.saturation),
            int(self.brightness),
        )
        upper = (min(179, blue_hue + h), 255, 255)

        mask = cv2.inRange(hsv, lower, upper)

        
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
                    and 60 <= g['internal_angle_deg'] <= 120):
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

    def check_ready(self, frame, now):
        """Update readiness from CURRENT points; old tracked positions never count."""
        self.ready = False
        g = measure(self.points) if self.points is not None else None
        if g is None:
            self.aligned_since = None
            self.calibration_status = 'Aligning: markers not tracked'
            return
        height, width = frame.shape[:2]
        center_error = np.linalg.norm(self.points['center'] - [width/2, height/2])
        checks = {
            'rotation': abs(g['alpha_deg']) <= ROTATION_TOLERANCE_DEG,
            'centering': center_error <= CENTER_TOLERANCE_PX,
            'right angle': abs(g['internal_angle_deg']-90) <= RIGHT_ANGLE_TOLERANCE_DEG,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            self.aligned_since = None
            self.calibration_status = 'Aligning: ' + ', '.join(failed)
            return
        if self.aligned_since is None:
            self.aligned_since = now
        elapsed = now - self.aligned_since
        self.ready = bool(elapsed >= READY_HOLD_SECONDS)
        self.calibration_status = ('Ready' if self.ready else
                                  f'Hold position: {elapsed:.1f}/{READY_HOLD_SECONDS:.1f} s')

    def clear_distance_cache(self):
        """Call after changing the camera setup or marker placement."""
        self._distance_cache.clear()
        self._fresh_distances.clear()

    def get_line_distances(self):
        """Return latest distances plus freshness metadata; does not capture a frame.

        Each entry has pixels (None before first observation), cached,
        age_seconds, and updated_at (time.perf_counter clock).
        Call read() regularly to refresh. Returned dictionaries are copies.
        """
        now = time.perf_counter()
        result = {}
        for name in ('top_center', 'center_right'):
            pixels, updated = self._distance_cache.get(name, (None, None))
            result[name] = dict(
                pixels=pixels,
                cached=pixels is not None and name not in self._fresh_distances,
                age_seconds=None if updated is None else max(0., now-updated),
                updated_at=updated,
            )
        return result

    def get_alpha(self):
        geometry = measure(self.points) if self.points is not None else None
        return geometry['alpha_deg'] if geometry is not None else None

    def get_top_center_distance(self):
        """Latest top-centre pixels, or None until first measured."""
        return self._distance_cache.get('top_center', (None, None))[0]

    def get_center_right_distance(self):
        """Latest centre-right pixels, or None until first measured."""
        return self._distance_cache.get('center_right', (None, None))[0]

    def _update_line_distances(self, anchors, now):
        # Full tracking gives current labelled points. During short occlusions,
        # independently match still-visible markers to the previous identities.
        visible = self.points
        if visible is None and anchors is not None:
            matches = {name: [i for i, p in enumerate(self.candidates)
                             if np.linalg.norm(p-anchors[name]) <= MAX_STEP_PX]
                       for name in NAMES}
            unique = {name: ids[0] for name, ids in matches.items() if len(ids) == 1}
            visible = {name: self.candidates[i] for name, i in unique.items()
                       if list(unique.values()).count(i) == 1}
        if visible is None:
            return
        for name, a, b in [('top_center', 'top', 'center'),
                            ('center_right', 'center', 'right')]:
            if a in visible and b in visible:
                pixels = float(np.linalg.norm(visible[a]-visible[b]))
                if pixels >= 1:
                    self._distance_cache[name] = (pixels, now)
                    self._fresh_distances.add(name)

    def read(self):
        """Return (BGR frame, blue mask, current points or None, ready bool)."""
        self._fresh_distances.clear()
        ok, frame = self.capture.read()
        now = time.perf_counter()
        if not ok:
            return None
        settings = (self.hue, self.saturation, self.brightness)
        if settings != self.settings:
            self.reset()
            self.settings = settings
        self.candidates, mask = self.detect(frame)
        anchors = self.last_points
        self.track(self.candidates)
        self._update_line_distances(anchors, now)
        self.check_ready(frame, now)
        return frame, mask, self.points, self.ready

    def close(self):
        """Release the camera when the PROJECT is finished using it."""
        self.capture.release()


class CameraPreview:
    """Optional test UI. show() returns True on Escape, NOT on readiness.

    R resets calibration. close() destroys only these two preview windows,
    leaving the sensor camera open for a project's next stage.
    """
    CAMERA_WINDOW = 'Platform camera'
    SETTINGS_WINDOW = 'Marker settings'

    def __init__(self, sensor):
        self.sensor = sensor
        self.closed = False
        try:
            cv2.namedWindow(self.CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.SETTINGS_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.CAMERA_WINDOW, 20, 60)
            cv2.moveWindow(self.SETTINGS_WINDOW, 940, 60)
            for label, attr, maximum in [('H', 'hue', 30), ('S', 'saturation', 255), ('V', 'brightness', 255)]:
                cv2.createTrackbar(label, self.SETTINGS_WINDOW, getattr(sensor, attr), maximum,
                                  lambda value, attr=attr: setattr(sensor, attr,
                                      max(1, value) if attr == 'hue' else value))
        except Exception:
            self.close()
            raise

    @staticmethod
    def fit(image, width, height):
        scale = min(width/image.shape[1], height/image.shape[0])
        return cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                                  max(1, round(image.shape[0]*scale))),
                          interpolation=cv2.INTER_NEAREST)

    def show(self, frame, mask, points, ready):
        """Display one sample without modifying the caller's frame. True = quit."""
        if self.closed:
            raise RuntimeError('Preview is closed')
        image = frame.copy()
        height, width = image.shape[:2]
        target = (width//2, height//2)
        cv2.drawMarker(image, target, (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
        cv2.circle(image, target, round(CENTER_TOLERANCE_PX), (255, 255, 0), 1)
        for p in self.sensor.candidates:
            cv2.circle(image, tuple(np.rint(p).astype(int)), 10, (0, 255, 255), 2)
        if points is not None:
            center = tuple(np.rint(points['center']).astype(int))
            for name in NAMES:
                pixel = tuple(np.rint(points[name]).astype(int))
                cv2.line(image, center, pixel, (0, 0, 0), 1)
                cv2.putText(image, name, (pixel[0]+12, pixel[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        image = self.fit(image, 900, 600)
        distances = self.sensor.get_line_distances()
        alpha = self.sensor.get_alpha()
        alpha_text = '--' if alpha is None else f'{alpha:+.2f}'
        lines = [f'Ready: {ready}', self.sensor.calibration_status,
                 f'Alpha: {alpha_text} deg']
        for name, label in [('top_center', 'Top-Center Distance'),
                            ('center_right', 'Center-Right Distance')]:
            entry = distances[name]
            value = '--' if entry['pixels'] is None else f"{entry['pixels']:.1f}"
            suffix = ' (cached)' if entry['cached'] else ''
            lines.append(f'{label}: {value} px{suffix}')
        lines += ['R: reset | Esc: quit']
        for i, text in enumerate(lines):
            cv2.putText(image, text, (12, 24+24*i), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 255, 0) if ready else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.CAMERA_WINDOW, image)
        labels = np.full((90, 420, 3), 245, np.uint8)
        descriptions = [f'H: blue hue tolerance = {self.sensor.hue}',
                        f'S: Minimum saturation = {self.sensor.saturation}',
                        f'V: Minimum brightness = {self.sensor.brightness}']
        for i, text in enumerate(descriptions):
            cv2.putText(labels, text, (10, 23+27*i), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (0, 0, 0), 1, cv2.LINE_AA)
        mask_preview = self.fit(mask, 420, 270)
        canvas = np.zeros((mask_preview.shape[0], 420), np.uint8)
        left = (420-mask_preview.shape[1])//2
        canvas[:, left:left+mask_preview.shape[1]] = mask_preview
        cv2.imshow(self.SETTINGS_WINDOW, np.vstack((labels, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.sensor.reset()
        return key == 27

    def close(self):
        if not self.closed:
            for name in (self.CAMERA_WINDOW, self.SETTINGS_WINDOW):
                try:
                    cv2.destroyWindow(name)
                except cv2.error:
                    pass
            self.closed = True
