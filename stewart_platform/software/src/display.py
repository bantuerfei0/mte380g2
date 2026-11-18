"""Shared OpenCV window with a clickable side panel."""
import textwrap
import cv2
import numpy as np


def label(image, text, xy, color=(230, 230, 230)):
    cv2.putText(image, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, .49, color, 1, cv2.LINE_AA)


class Window:
    def __init__(self, title):
        self.title = title
        self.click = None
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def poll(self, image, buttons=(), status=""):
        """Return (selected button, image click, key). Closing the window produces Escape."""
        width = image.shape[1]
        if buttons or status:
            lines = [part for line in status.splitlines() for part in textwrap.wrap(line, 29)]
            height = max(image.shape[0], len(buttons)*28 + len(lines)*20 + 12)
            canvas = np.full((height,width+265,3), 31, np.uint8)
            canvas[:image.shape[0], :width] = image
            for i, button in enumerate(buttons):
                cv2.rectangle(canvas, (width+3,28*i+1), (width+262,28*i+26), (62,62,62), -1)
                label(canvas, button, (width+10,28*i+19))
            for i, line in enumerate(lines):
                label(canvas, line, (width+8,28*len(buttons)+20*i+17))
        else:
            canvas = image
        cv2.imshow(self.title, canvas)
        key = cv2.waitKey(12) & 255
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return None, None, 27
        click, self.click = self.click, None
        if click and click[0] >= width:
            row = click[1]//28
            return buttons[row] if row < len(buttons) else None, None, key
        return None, click, key

    def close(self):
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass  # Some backends already destroy a closed window.
