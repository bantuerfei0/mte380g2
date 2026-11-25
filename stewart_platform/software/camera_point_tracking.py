"""Blue-marker calibration for an EXISTING camera loop.

sensor = CameraSensor()                 # Does not open a camera or any windows.
ready = sensor.process(frame)           # Pass each original BGR camera frame.
alpha = sensor.get_alpha()              # Current image rotation, or None.
top_px = sensor.get_top_center_distance() # Last valid pixels, even if obscured.
right_px = sensor.get_center_right_distance()

Optional: preview = CameraPreview(sensor); preview.show(frame); preview.close().
All distances use original image pixels. No physical-distance checks are made.
"""
import time
import cv2
import numpy as np

# Fixed BLUE thresholds from your uploaded file: hue 110 +/-19, S>=133, V>=54.
BLUE_LOWER = (91, 133, 54)
BLUE_UPPER = (129, 255, 255)
NAMES = ('top', 'center', 'right')
MAX_MISSED_FRAMES = 5
MAX_STEP_PX = 35
ROTATION_TOLERANCE_DEG = 1.0
RIGHT_ANGLE_TOLERANCE_DEG = 2.0
CENTER_TOLERANCE_PX = 10.0
READY_HOLD_SECONDS = 1.0


def measure(points):
    """Measure signed camera-relative rotation, internal angle and pixel distances."""
    top, center, right = (points[name] for name in NAMES)
    u, v = top-center, right-center
    tc, rc = np.linalg.norm(u), np.linalg.norm(v)
    if min(tc, rc) < 1:
        return None
    alpha = np.degrees(np.arccos(np.clip(np.dot(u, v)/(tc*rc), -1, 1)))
    return dict(alpha_deg=float(np.degrees(np.arctan2(u[0], -u[1]))),
                internal_angle_deg=float(alpha), distance_tc_px=float(tc),
                distance_rc_px=float(rc))


class CameraSensor:
    """Stateful frame processor; caller owns capture and the main loop."""

    def __init__(self):
        self._distance_cache = {}
        self._fresh_distances = set()
        self.candidates, self.mask = [], None
        self.reset()

    def reset(self):
        self._fresh_distances.clear()  # Retain distances through tracking resets.
        self.points = None
        self.last_points, self.missed = None, 0
        self.status = 'Searching: return to starting orientation'
        self.ready, self.aligned_since = False, None
        self.calibration_status = 'Aligning'

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
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
        """Match by proximity; keep identities through five incomplete frames."""
        if self.last_points is None:
            visible = self.label(dots) or {}
        else:
            matches = {name: [i for i, p in enumerate(dots)
                             if np.linalg.norm(p-self.last_points[name]) <= MAX_STEP_PX]
                       for name in NAMES}
            unique = {name: ids[0] for name, ids in matches.items() if len(ids) == 1}
            visible = {name: dots[i] for name, i in unique.items()
                       if list(unique.values()).count(i) == 1}
        self.points = visible if len(visible) == 3 else None
        if self.points is not None:
            self.last_points, self.missed = self.points, 0
        else:
            self.missed += 1
            if self.missed > MAX_MISSED_FRAMES:
                self.last_points = None
        return visible  # Partial matches can still update an unobscured line.

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
        """Clear pixel references after changing the physical camera setup."""
        self._distance_cache.clear()
        self._fresh_distances.clear()

    def get_alpha(self):
        """Current rotation in degrees; None when any marker is missing."""
        geometry = measure(self.points) if self.points is not None else None
        return geometry['alpha_deg'] if geometry is not None else None

    def get_top_center_distance(self):
        """Latest top-centre pixels, cached when obscured; None before detection."""
        return self._distance_cache.get('top_center', (None, None))[0]

    def get_center_right_distance(self):
        """Latest centre-right pixels, cached when obscured; None before detection."""
        return self._distance_cache.get('center_right', (None, None))[0]

    def get_line_distances(self):
        """Return pixel values and whether each came from a previous frame."""
        now = time.perf_counter()
        return {name: dict(pixels=value, cached=value is not None and name not in self._fresh_distances,
                           age_seconds=None if stamp is None else max(0., now-stamp))
                for name in ('top_center', 'center_right')
                for value, stamp in [self._distance_cache.get(name, (None, None))]}

    def process(self, frame):
        """Process one BGR frame and return ready. No capture or GUI calls.

        ready requires all alignment checks for one continuous second. Passing
        None clears readiness after a capture failure but keeps cached distances.
        """
        self._fresh_distances.clear()
        if frame is None:
            self.reset()
            self.candidates, self.mask = [], None
            return False
        now = time.perf_counter()
        self.candidates, self.mask = self.detect(frame)
        visible = self.track(self.candidates)
        # Update each line independently; missing endpoints leave its cache intact.
        for name, a, b in [('top_center', 'top', 'center'), ('center_right', 'center', 'right')]:
            if a in visible and b in visible:
                self._distance_cache[name] = (float(np.linalg.norm(visible[a]-visible[b])), now)
                self._fresh_distances.add(name)
        self.check_ready(frame, now)
        return self.ready


class CameraPreview:
    """Optional visual helper. show(frame) returns True ONLY for Escape.

    Closing this helper does not touch the application's camera.
    """
    CAMERA_WINDOW = 'Platform calibration'
    MASK_WINDOW = 'Blue marker mask'

    def __init__(self, sensor):
        self.sensor = sensor
        cv2.namedWindow(self.CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.MASK_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.CAMERA_WINDOW, 20, 60)
        cv2.moveWindow(self.MASK_WINDOW, 940, 60)

    @staticmethod
    def fit(image, width, height):
        scale = min(width/image.shape[1], height/image.shape[0])
        size = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

    def show(self, frame):
        # Draw on a copy: other project code can still use the original frame.
        sensor = self.sensor
        image = frame.copy()
        height, width = image.shape[:2]
        target = (width//2, height//2)
        cv2.drawMarker(image, target, (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
        cv2.circle(image, target, round(CENTER_TOLERANCE_PX), (255, 255, 0), 1)
        for point in sensor.candidates:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 10, (0, 255, 255), 2)
        if sensor.points is not None:
            center = tuple(np.rint(sensor.points['center']).astype(int))
            for name in NAMES:
                point = tuple(np.rint(sensor.points[name]).astype(int))
                cv2.line(image, center, point, (0, 0, 0), 1)
                cv2.putText(image, name, (point[0]+12, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        image = self.fit(image, 900, 600)
        alpha = sensor.get_alpha()
        lines = [f'Ready: {sensor.ready}', sensor.calibration_status,
                 'Alpha: --' if alpha is None else f'Alpha: {alpha:+.2f} deg']
        for name, label in [('top_center', 'Top-Center'), ('center_right', 'Center-Right')]:
            value = sensor.get_line_distances()[name]
            text = '--' if value['pixels'] is None else f"{value['pixels']:.1f}"
            cached = ' (cached)' if value['cached'] else ''
            lines.append(f'{label} Distance: {text} px{cached}')
        lines.append('Esc: quit')
        for i, text in enumerate(lines):
            cv2.putText(image, text, (12, 24+24*i), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 255, 0) if sensor.ready else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.CAMERA_WINDOW, image)
        if sensor.mask is not None:
            cv2.imshow(self.MASK_WINDOW, self.fit(sensor.mask, 420, 270))
        return cv2.waitKey(1) & 0xFF == 27

    def close(self):
        for window in (self.CAMERA_WINDOW, self.MASK_WINDOW):
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
