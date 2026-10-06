from src.abs.position_sensor import PositionSensor
import cv2
import numpy as np
import threading


class CameraSensor(PositionSensor):
    """
    Camera-based ball sensor
    """
    def __init__(self, config: list) -> None:
        super().__init__()
        # assume is only camera
        self.cam = cv2.VideoCapture(0, cv2.CAP_V4L2, config)
        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera 0")
        self.frame: np.ndarray | None = None
        self.done = threading.Event()
        self.has_new = False
        self.thread = threading.Thread(target=self.task, daemon=True)

    def get_position(self):
        # do a shit ton of processing
        pass

    def task(self):
        while not self.done.is_set():
            ok, frame = self.cam.read()
            if not ok:
                # if there was no frame
                continue
            self.frame, self.has_new = frame, True
    
    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=1)
        self.cam.release()

if __name__ == "__main__":
    # testing code  
    camera_settings = [
        cv2.CAP_PROP_FRAME_WIDTH, 640,
        cv2.CAP_PROP_FRAME_HEIGHT, 480,
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_AUTO_WB, 0,
        cv2.CAP_PROP_AUTO_EXPOSURE,1,
        cv2.CAP_PROP_GAIN, 0,
        cv2.CAP_PROP_EXPOSURE, 300
    ]
    camera_sensor = CameraSensor(camera_settings)
    camera_sensor.start()
    while True:
        frame = camera_sensor.frame
        if frame is not None:
            cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()