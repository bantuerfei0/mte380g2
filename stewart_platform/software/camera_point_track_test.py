"""Example only: replace this capture block with your project's existing camera."""
import cv2
from camera_point_tracking import CameraSensor, CameraPreview


def main():
    sensor = CameraSensor()  # Processing only: does not open a camera.
    capture = cv2.VideoCapture(1)  # Demo camera; your integrated code already has one.
    preview = None
    started = False
    try:
        preview = CameraPreview(sensor)
        while True:
            ok, frame = capture.read()  # Capture ONCE, then share this same frame.
            ready = sensor.process(frame if ok else None)
            if not ok or frame is None:
                if cv2.waitKey(50) & 0xFF == 27:
                    break
                continue
            if preview.show(frame):
                break
            if ready and not started:
                print('Calibration ready: True')
                started = True  # Trigger the next stage here once; keep preview running.
            # Other code can use sensor.get_alpha() and both distance getters here.
    finally:
        if preview is not None:
            preview.close()
        capture.release()  # Only the camera owner releases the device.


if __name__ == '__main__':
    main()
