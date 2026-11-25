import argparse
from camera_point_tracking import CameraSensor, CameraPreview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    sensor = CameraSensor(device_index=parser.parse_args().camera)
    preview = None
    next_step_started = False
    try:
        preview = CameraPreview(sensor)
        while True:
            frame, mask, points, ready = sensor.read()

            print(
                f"Top-Center: {sensor.get_top_center_distance()} px | "
                f"Center-Right: {sensor.get_center_right_distance()} px"
            )
            
            if preview.show(frame, mask, points, ready):
                break  # Escape cancels; do not start the next step.
            if ready and sensor.ready and not next_step_started:
                print('Calibration ready: True. The next project step can begin.')
                next_step_started = True
                # Trigger the next stage here ONCE. Keep long-running control
                # work out of this loop so the camera preview remains responsive.
                # This tester keeps monitoring ready until Escape is pressed.
    finally:
        if preview is not None:
            preview.close()
        sensor.close()  # Release only after the next stage has finished.


if __name__ == '__main__':
    main()
