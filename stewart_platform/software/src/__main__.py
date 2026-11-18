"""Click the plate centre, click the ball, then press Space to balance."""
import argparse
import os
import time
from contextlib import ExitStack

def find_ball(hsv, colour):
    import cv2
    import numpy as np

    hue = hsv[:, :, 0].astype(np.int16)
    hue_distance = np.abs((hue - int(colour[0]) + 90) % 180 - 90)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    matches = ((hue_distance <= 12) | (colour[1] < 35))
    matches &= (np.abs(saturation - int(colour[1])) <= 75)
    matches &= (np.abs(value - int(colour[2])) <= 75)
    mask = cv2.morphologyEx(matches.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if 20 <= area <= 8000 and perimeter and 4 * np.pi * area / perimeter**2 > .6:
            moments = cv2.moments(contour)
            candidates.append((area, np.array([moments["m10"], moments["m01"]]) / area))
    return max(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


def balance(camera, write, park, inverse):
    import cv2
    import numpy as np

    title = "Ball balancer"
    target = colour = hsv = previous = None
    velocity = np.zeros(2)
    armed = False

    def click(event, x, y, flags, data):
        nonlocal target, colour
        if (event != cv2.EVENT_LBUTTONDOWN or armed or hsv is None
                or not (0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1])):
            return
        if target is None:
            target = np.array([x, y])
        else:
            colour = hsv[y, x].copy()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, click)
    last_time = time.monotonic()
    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera frame unavailable")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball = find_ball(hsv, colour) if colour is not None else None
        now = time.monotonic()
        if armed and ball is None:
            armed = False
            write(park)
        if armed:
            if previous is not None:
                velocity = .7 * velocity + .3 * (ball - previous) / max(now - last_time, .001)
            tilt = (.018 * (target - ball) - .012 * velocity) * (640 / frame.shape[1])
            tilt *= min(1, 3 / max(np.linalg.norm(tilt), .001))
            write(park + inverse @ tilt)
        else:
            velocity.fill(0)
        previous, last_time = ball if armed else None, now
        if target is not None:
            cv2.drawMarker(frame, tuple(target), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if ball is not None:
            cv2.circle(frame, tuple(ball.astype(int)), 8, (255, 255, 0), 2)
        status = ("Balancing | Space disarm" if armed else
                  "Click plate centre" if target is None else
                  "Click ball to sample | Space arm" if ball is not None else
                  "Click ball to sample")
        cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xff
        if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            return
        if key == ord(" ") and (armed or ball is not None):
            armed = not armed
            if not armed:
                write(park)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--park", type=float, nargs=3, required=True,
                        help="verified level pulse widths in us, channel order 2 0 3")
    parser.add_argument("--axes", type=int, choices=(-3, -2, -1, 1, 2, 3),
                        nargs=2, required=True,
                        help="signed IMU axes for downhill toward image right and down")
    parser.add_argument("--probe", type=float, default=60, help="probe half-sweep in us")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--chip", type=int, default=0)
    args = parser.parse_args()
    import cv2
    import numpy as np

    from .imu import open_imu
    from .servos import MIN_US, MAX_US, open_servos, probe_servos

    park = np.array(args.park)
    if (not np.isfinite(args.probe) or args.probe <= 0 or
            not np.all((park - args.probe >= MIN_US) & (park + args.probe <= MAX_US))):
        parser.error("Each park pulse plus and minus the probe must be within 1600–2480 us")
    if abs(args.axes[0]) == abs(args.axes[1]):
        parser.error("The two IMU axes must be distinct")
    if os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    with ExitStack() as resources:
        camera = cv2.VideoCapture(args.camera)
        resources.callback(camera.release)
        resources.callback(cv2.destroyAllWindows)
        if not camera.isOpened():
            raise RuntimeError("Could not open camera")
        write = resources.enter_context(open_servos(park, args.chip))
        print("Probing servos. Keep the ball off the plate.", flush=True)
        with open_imu(np.array(args.axes)) as read_tilt:
            inverse = probe_servos(write, read_tilt, park, args.probe)
        balance(camera, write, park, inverse)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
