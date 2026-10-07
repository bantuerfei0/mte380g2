"""Entry point. Run from the ball_balancer folder:  python -m src.main"""

import os
from pathlib import Path

# The opencv-python wheel bundles Qt without any fonts, which triggers a QFontDatabase warning.
# Point Qt at the system fonts; this must happen before cv2 is imported.
for font_dir in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"):
    if Path(font_dir).is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", font_dir)
        break

from src.ball_balancer import BallBalancer  # noqa: E402  (must come after the font setup)

if __name__ == "__main__":
    BallBalancer().run()
