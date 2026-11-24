import argparse
from camera_point_tracking import CameraSensor, CameraPreview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1) # default= 0 for webcam, 1 for camera
    sensor = CameraSensor(device_index=parser.parse_args().camera)
    preview = None
    try:
        preview = CameraPreview(sensor)
        while True:
            frame, mask, points, ready = sensor.read()
            if preview.show(frame, mask, points, ready):
                break  # Escape cancels; do not start the next step.
            if ready and sensor.ready:  # R can reset readiness in show().
                print('Calibration ready: True. The next project step can begin.')
                # Call your next project stage HERE, passing sensor if it needs
                # the same camera. This executes once because the loop exits.
                break
    finally:
        if preview is not None:
            preview.close()
        sensor.close()  # Release only after the next stage has finished.


if __name__ == '__main__':
    main()
