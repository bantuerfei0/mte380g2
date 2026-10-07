"""
Plots a recorded response:  python tools/plot_recording.py recordings/<file>.csv
Needs matplotlib (only this script uses it).
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
axis_ids = sorted({column[-1] for column in data.dtype.names if column != "t"})

figure, plot_rows = plt.subplots(len(axis_ids), 2, sharex=True, squeeze=False, figsize=(12, 4 * len(axis_ids)))
for (response_plot, terms_plot), axis in zip(plot_rows, axis_ids):
    for signal in ("target", "position", "command"):
        response_plot.plot(data["t"], data[signal + axis], label=signal)
    for term in ("P", "I", "D"):
        terms_plot.plot(data["t"], data[term + axis], label=term)
    response_plot.set_title(f"axis {axis}: response")
    terms_plot.set_title(f"axis {axis}: PID terms")
    for plot in (response_plot, terms_plot):
        plot.grid(True)
        plot.legend()
for plot in plot_rows[-1]:
    plot.set_xlabel("t (s)")
plt.tight_layout()
plt.show()
