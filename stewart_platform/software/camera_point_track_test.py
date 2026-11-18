import argparse
import cv2
import numpy as np
from camera_point_tracking import CameraSensor, NAMES
CAMERA_WIDTH = 900
SETTINGS_WIDTH = 420


def fit_image(image, width, max_height, interpolation=cv2.INTER_AREA):
    scale = min(width/image.shape[1], max_height/image.shape[0])
    return cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                              max(1, round(image.shape[0]*scale))), interpolation=interpolation)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    sensor = CameraSensor(parser.parse_args().camera)
    try:
        # Autosize follows the rendered images instead of retaining oversized windows.
        cv2.namedWindow('Platform camera', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Marker settings', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('Platform camera', 20, 60)
        cv2.moveWindow('Marker settings', CAMERA_WIDTH+40, 60)
        for label, attr, maximum in [('H', 'hue', 30), ('S', 'saturation', 255), ('V', 'brightness', 255)]:
            cv2.createTrackbar(label, 'Marker settings', getattr(sensor, attr), maximum,
                              lambda value, attr=attr: setattr(sensor, attr, max(1, value) if attr == 'hue' else value))
        while True:
            frame, mask, points, result = sensor.read()
            for p in sensor.candidates:
                cv2.circle(frame, tuple(np.rint(p).astype(int)), 10, (0, 255, 255), 2)
            if points:
                center = tuple(np.rint(points['center']).astype(int))
                for name in NAMES:
                    p = tuple(np.rint(points[name]).astype(int))
                    cv2.line(frame, center, p, (0, 0, 0), 1)
                    cv2.putText(frame, name, (p[0]+12, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
            preview = fit_image(frame, CAMERA_WIDTH, 600)
            lines = [sensor.status]
            if result:
                lines += [f"Alpha: {result['alpha_deg']:.2f} deg",
                          f"Top-Center Distance: {result['distance_tc_px']:.1f} px",
                          f"Right-Center Distance: {result['distance_rc_px']:.1f} px"]
            else:
                lines += ['Alpha: --', 'Top-Center Distance: --', 'Right-Center Distance: --']
            lines += ['R: reset | Esc: quit']
            for i, text in enumerate(lines):
                cv2.putText(preview, text, (12, 24+24*i), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Platform camera', preview)
            labels = np.full((90, SETTINGS_WIDTH, 3), 245, np.uint8)
            for i, text in enumerate([f'H: Red hue tolerance = {sensor.hue}',
                                      f'S: Minimum saturation = {sensor.saturation}',
                                      f'V: Minimum brightness = {sensor.brightness}']):
                cv2.putText(labels, text, (10, 23+27*i), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, (0, 0, 0), 1, cv2.LINE_AA)
            mask_preview = fit_image(mask, SETTINGS_WIDTH, 270, cv2.INTER_NEAREST)
            # Centre unusual aspect ratios on a compact canvas.
            canvas = np.zeros((mask_preview.shape[0], SETTINGS_WIDTH), np.uint8)
            left = (SETTINGS_WIDTH-mask_preview.shape[1])//2
            canvas[:, left:left+mask_preview.shape[1]] = mask_preview
            cv2.imshow('Marker settings', np.vstack((labels, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('r'):
                sensor.reset()
    finally:
        sensor.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
