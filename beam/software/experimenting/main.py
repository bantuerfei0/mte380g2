import threading
import time

import serial

PORT = "COM3"        # e.g. "/dev/ttyACM0" on Linux, "/dev/cu.usbmodemXXXX" on macOS
BAUD = 115200          # must match Serial.begin() on the Mega
NUM_SENSORS = 8


class SensorReader:
    """Reads 2-byte frames from the Mega in a background thread.

    Frame format:
        byte 1: 1 iii 000 v   (MSB=1, 3-bit index, top bit of value)
        byte 2: 0 vvvvvvv     (MSB=0, low 7 bits of value)
    """

    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.values = [0] * NUM_SENSORS
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.ser.close()

    def get_values(self):
        with self.lock:
            return list(self.values)  # copy so the caller gets a stable snapshot

    def _run(self):
        first = None  # holds the first byte of a frame until its partner arrives
        while not self._stop.is_set():
            data = self.ser.read(self.ser.in_waiting or 1)  # read in chunks
            for b in data:
                if b & 0x80:                 # start of a new frame
                    first = b
                elif first is not None:      # second byte of a frame
                    idx = (first >> 4) & 0x07
                    val = ((first & 0x01) << 7) | b
                    with self.lock:
                        self.values[idx] = val
                    first = None
                # a second byte with no first byte (e.g. right after connecting) is dropped


if __name__ == "__main__":
    reader = SensorReader(PORT, BAUD)
    reader.start()
    try:
        while True:
            print(reader.get_values())
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()