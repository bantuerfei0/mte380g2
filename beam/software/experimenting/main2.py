import threading
import time

import serial

import pygame

PORT = "COM3"        # e.g. "/dev/ttyACM0" on Linux, "/dev/cu.usbmodemXXXX" on macOS
BAUD = 115200          # must match Serial.begin() on the Mega
NUM_SENSORS = 8

pygame.init()

W, H = 800, 400
STEPS = 16  # line segments between each pair of points
 
 
def draw_curve(screen, values):
    """Smooth Catmull-Rom curve through 8-bit values (0-255), spread across the window."""
    n = len(values)
    dx = W / (n - 1)
    pts = []
    for i in range(n - 1):
        p0, p1, p2, p3 = values[max(i - 1, 0)], values[i], values[i + 1], values[min(i + 2, n - 1)]
        for s in range(STEPS + 1):
            t = s / STEPS
            v = 0.5 * (2*p1 + (p2 - p0)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t**2 + (3*p1 - p0 - 3*p2 + p3)*t**3)
            pts.append(((i + t) * dx, H - v * H / 255 + H / 2))
    pygame.draw.lines(screen, (80, 200, 255), False, pts, 2)

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
    main_screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock() # used for various time features, but mainly for keeping a steady framerate (see later)
    # sets the window title
    pygame.display.set_caption('visualiser')
    done = False
    try:
        while not done:
            for event in pygame.event.get():
                # UNCOMMENT THIS IF YOU WANT TO SEE WHAT EVENTS ARE BEING BROUGHT IN
                #print(event)
                
                # if the event's type is pygame.QUIT (a constant defined in pygame)
                if event.type == pygame.QUIT:
                    done = True # this causes pygame to exit on the next loop
            main_screen.fill((20, 20, 28))
            values = reader.get_values()
            draw_curve(main_screen, values)
            print(values)
            pygame.display.flip() # redraws the frame on the main window with what is on main_screen
            
            # pause the program long enough to hit 60 fps
            clock.tick(60) # pauses the program until it should run again to keep a steady framerate
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()