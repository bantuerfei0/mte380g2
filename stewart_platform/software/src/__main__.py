"""Launch using python -m src from stewart_platform/software."""
import argparse
from pathlib import Path

from .settings import CONFIG


def main():
    parser = argparse.ArgumentParser(prog="python -m src", description="Stewart platform ball balancer")
    parser.add_argument("mode", nargs="?", choices=["preview", "measure", "identify", "home", "continuous",
                                                     "simulate", "analyze"])
    parser.add_argument("recordings", nargs="*", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.mode in ("simulate", "analyze"):
        from .experiments import report
        if bool(args.recordings) != (args.mode == "analyze"):
            parser.error("analyze requires recording filenames; simulate takes none")
        print("\n".join(report(args.recordings, plot=args.plot)))
    elif args.mode in ("measure", "identify"):
        from contextlib import ExitStack
        from threading import Event
        from .calibration import configure
        from .settings import load
        with ExitStack() as hardware:
            action = "read" if args.mode == "measure" else "identify"
            _, _, message = configure(action, load(args.config), args.config, hardware, Event())
            print(message)
    else:
        from .gui import launch
        launch(args.config, args.mode)


if __name__ == "__main__":
    main()
