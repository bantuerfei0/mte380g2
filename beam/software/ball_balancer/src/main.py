import os
from pathlib import Path

# The opencv-python wheel bundles Qt without fonts; point Qt at the system fonts before cv2 loads.
for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"):
    if Path(d).is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", d)
        break

from src.ball_balancer import BallBalancer  # noqa: E402

if __name__ == "__main__":
    BallBalancer().run()
