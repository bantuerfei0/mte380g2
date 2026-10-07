"""Plot a recording: python tools/plot_recording.py recordings/<file>.csv  (needs matplotlib)."""
import sys

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
axes = sorted({n[-1] for n in data.dtype.names if n != "t"})
fig, plots = plt.subplots(len(axes), 2, sharex=True, squeeze=False, figsize=(12, 4 * len(axes)))
for row, a in zip(plots, axes):
    for f in ("target", "pos", "out"):
        row[0].plot(data["t"], data[f + a], label=f)
    for f in ("P", "I", "D"):
        row[1].plot(data["t"], data[f + a], label=f)
    row[0].set_title(f"axis {a}: response")
    row[1].set_title(f"axis {a}: PID terms")
    for ax in row:
        ax.grid(True)
        ax.legend()
for ax in plots[-1]:
    ax.set_xlabel("t (s)")
plt.tight_layout()
plt.show()
