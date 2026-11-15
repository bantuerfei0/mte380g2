"""
The application window: the active sensor's image on the left, a control panel on the right.

Everything is drawn with cv2.putText/rectangle, so it looks the same on every OpenCV GUI
backend and never depends on Qt widgets or Qt fonts. Only imshow, waitKey and the mouse
callback are used from HighGUI.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Panel colours (BGR)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DARK_GREY = (60, 60, 60)
PANEL_BACKGROUND = (30, 30, 30)
ACCENT = (0, 220, 255)

# Colours shared by every view, so markers and plot lines mean the same thing everywhere (BGR)
COLOR_TARGET = (0, 0, 255)       # red: target position
COLOR_BALL = (0, 255, 0)         # green: measured ball position
COLOR_COMMAND = (0, 220, 255)    # yellow: tilt command sent to the servo
COLOR_REFERENCE = (255, 255, 0)  # cyan: beam line, baselines, thresholds

KEY_ENTER = (10, 13)
KEY_ESCAPE = 27
KEY_BACKSPACE = (8, 127)


def draw_text(image: np.ndarray, message: str, origin: tuple[int, int], color=WHITE, scale: float = 0.45) -> None:
    cv2.putText(image, message, origin, FONT, scale, color, 1, cv2.LINE_AA)


@dataclass
class Button:
    label: str | Callable[[], str]  # a callable label is re-evaluated every frame, e.g. "Record" / "Stop rec"
    on_click: Callable[[], None]
    group: str
    rect: tuple = (0, 0, 0, 0)  # window coordinates (x0, y0, x1, y1), set while drawing


@dataclass
class Slider:
    """A horizontal slider bound directly to `target_object.attribute`."""

    label: str
    target_object: object
    attribute: str
    minimum: float
    maximum: float
    group: str
    bar_rect: tuple = (0, 0, 0, 0)    # click or drag here to set the value
    label_rect: tuple = (0, 0, 0, 0)  # click here to type an exact value

    @property
    def value(self) -> float:
        return getattr(self.target_object, self.attribute)

    @value.setter
    def value(self, new_value: float) -> None:
        setattr(self.target_object, self.attribute, new_value)

    def set_from_mouse_x(self, x: int) -> None:
        x0, _, x1, _ = self.bar_rect
        fraction = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
        self.value = self.minimum + fraction * (self.maximum - self.minimum)


class UI:
    """
    Widgets are tagged with a group so a component can remove its own controls later
    (e.g. a sensor's controls when switching sensors).

    Mouse: buttons are clicked; slider bars are clicked or dragged; clicking a slider's
    label lets you type an exact value (Enter applies, Esc cancels, values may exceed the
    slider range). Other clicks on the image go to the active click-capture handler, which
    sensors use for calibration; Enter cancels a capture.
    """

    PANEL_WIDTH = 260
    BUTTON_HEIGHT = 26
    PADDING = 8

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)
        self.widgets: list[Button | Slider] = []
        self.prompt_text = ""  # shown along the bottom of the image while capturing clicks
        self._current_group = ""
        self._dragged_slider: Slider | None = None
        self._edited_slider: Slider | None = None
        self._typed_text = ""
        self._click_handler: Callable[[int, int], None] | None = None
        self._image_width = 0

    # --- building the panel ---

    @contextmanager
    def widget_group(self, group: str):
        """Widgets added inside this block belong to `group`."""
        previous, self._current_group = self._current_group, group
        try:
            yield
        finally:
            self._current_group = previous

    def add_button(self, label, on_click: Callable[[], None]) -> None:
        self.widgets.append(Button(label, on_click, self._current_group))

    def add_slider(self, label: str, target_object, attribute: str, minimum: float, maximum: float) -> None:
        self.widgets.append(Slider(label, target_object, attribute, minimum, maximum, self._current_group))

    def remove_group(self, group: str) -> None:
        self.widgets = [widget for widget in self.widgets if widget.group != group]

    # --- click capture (used for calibration clicks on the image) ---

    def begin_click_capture(self, handler: Callable[[int, int], None], prompt: str) -> None:
        self._click_handler, self.prompt_text = handler, prompt

    def end_click_capture(self) -> None:
        self._click_handler, self.prompt_text = None, ""

    # --- input ---

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        def inside(rect) -> bool:
            return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]

        if event == cv2.EVENT_LBUTTONDOWN:
            self._edited_slider = None
            for widget in self.widgets:
                if isinstance(widget, Button) and inside(widget.rect):
                    widget.on_click()
                    return
                if isinstance(widget, Slider) and inside(widget.label_rect):
                    self._edited_slider, self._typed_text = widget, ""
                    return
                if isinstance(widget, Slider) and inside(widget.bar_rect):
                    self._dragged_slider = widget
                    widget.set_from_mouse_x(x)
                    return
            if self._click_handler and x < self._image_width:
                self._click_handler(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragged_slider and flags & cv2.EVENT_FLAG_LBUTTON:
            self._dragged_slider.set_from_mouse_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragged_slider = None

    def _handle_typing(self, key: int) -> None:
        """Keyboard input while a slider value is being typed."""
        if key in KEY_ENTER:
            try:
                self._edited_slider.value = float(self._typed_text)
            except ValueError:
                pass  # ignore malformed input and keep the old value
            self._edited_slider = None
        elif key == KEY_ESCAPE:
            self._edited_slider = None
        elif key in KEY_BACKSPACE:
            self._typed_text = self._typed_text[:-1]
        elif key != -1 and chr(key) in "0123456789.-e":
            self._typed_text += chr(key)

    # --- drawing ---

    def _draw_panel(self, status_lines: list[str]) -> np.ndarray:
        """Draws buttons, sliders and status text, recording each widget's window rect for clicks."""
        panel = np.full((2000, self.PANEL_WIDTH, 3), PANEL_BACKGROUND, np.uint8)  # cropped to content at the end
        offset_x, pad = self._image_width, self.PADDING  # panel x + offset_x = window x
        x, y = pad, pad

        for button in (w for w in self.widgets if isinstance(w, Button)):
            label = button.label() if callable(button.label) else button.label
            (text_width, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            button_width = text_width + 2 * pad
            if x + button_width > self.PANEL_WIDTH - pad:  # wrap to the next row
                x, y = pad, y + self.BUTTON_HEIGHT + 4
            cv2.rectangle(panel, (x, y), (x + button_width, y + self.BUTTON_HEIGHT), DARK_GREY, -1)
            cv2.rectangle(panel, (x, y), (x + button_width, y + self.BUTTON_HEIGHT), GREY, 1)
            draw_text(panel, label, (x + pad, y + 17))
            button.rect = (offset_x + x, y, offset_x + x + button_width, y + self.BUTTON_HEIGHT)
            x += button_width + 4
        y += self.BUTTON_HEIGHT + 12

        bar_left, bar_right = pad, self.PANEL_WIDTH - pad
        for slider in (w for w in self.widgets if isinstance(w, Slider)):
            if slider is self._edited_slider:
                draw_text(panel, f"{slider.label}: {self._typed_text}_", (pad, y + 12), ACCENT)
            else:
                draw_text(panel, f"{slider.label}: {slider.value:.4g}", (pad, y + 12))
            slider.label_rect = (offset_x, y, offset_x + self.PANEL_WIDTH, y + 16)

            bar_top, bar_bottom = y + 18, y + 28
            fraction = (slider.value - slider.minimum) / (slider.maximum - slider.minimum)
            fill_right = bar_left + int(min(max(fraction, 0.0), 1.0) * (bar_right - bar_left))
            cv2.rectangle(panel, (bar_left, bar_top), (bar_right, bar_bottom), DARK_GREY, -1)
            cv2.rectangle(panel, (bar_left, bar_top), (fill_right, bar_bottom), ACCENT, -1)
            slider.bar_rect = (offset_x + bar_left, bar_top - 4, offset_x + bar_right, bar_bottom + 4)
            y += 38
        y += 6

        for line in status_lines:
            draw_text(panel, line, (pad, y + 12))
            y += 18
        return panel[:y + pad]

    def show(self, image: np.ndarray, status_lines: list[str]) -> int:
        """
        Displays the image with the panel beside it and processes input.
        Returns the key pressed for the app to handle: -1 for none (or when the key was
        used for slider typing), Escape if the window was closed.
        """
        self._image_width = image.shape[1]
        if self.prompt_text:
            image_height = image.shape[0]
            cv2.rectangle(image, (0, image_height - 28), (self._image_width, image_height), (0, 0, 0), -1)
            draw_text(image, self.prompt_text, (8, image_height - 9), ACCENT, 0.55)

        panel = self._draw_panel(status_lines)
        window_height = max(image.shape[0], panel.shape[0])

        def pad_to_window_height(part: np.ndarray, fill) -> np.ndarray:
            extra = window_height - part.shape[0]
            return cv2.copyMakeBorder(part, 0, extra, 0, 0, cv2.BORDER_CONSTANT, value=fill)

        window = np.hstack((pad_to_window_height(image, (0, 0, 0)), pad_to_window_height(panel, PANEL_BACKGROUND)))
        cv2.imshow(self.window_name, window)

        key = cv2.waitKey(1)
        key = -1 if key == -1 else key & 0xFF
        if self._edited_slider:
            self._handle_typing(key)
            key = -1  # characters typed into a slider are not shortcuts
        elif key in KEY_ENTER and self._click_handler:
            self.end_click_capture()
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return KEY_ESCAPE
        return key
