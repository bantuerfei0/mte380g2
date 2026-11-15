# testing some random ass shit using clank clank
# referenced from https://jckantor.github.io/CBE30338/04.12-Interactive-PID-Control-Tuning-with-Ziegler--Nichols.html
import numpy as np
import matplotlib.pyplot as plt
import control


# ============================================================
# 1. PROCESS MODEL
# ============================================================

# Valve dynamics:
# Gv(s) = 1 / (2s + 1)

Gv = control.tf([1], [2, 1])


# Process dynamics:
# Gp(s) = 1 / (5s + 1)

Gp = control.tf([1], [5, 1])


# Disturbance dynamics:
# Gd(s) = 1 / (5s + 1)

Gd = control.tf([1], [5, 1])


# ============================================================
# 2. MEASUREMENT DELAY
# ============================================================

# Measurement delay:
# Gm(s) = e^(-s)
#
# Use a first-order Pade approximation.

tdelay = 1
pade_order = 1

num_delay, den_delay = control.pade(
    tdelay,
    pade_order
)

Gm = control.tf(
    num_delay,
    den_delay
)


# ============================================================
# 3. PI CONTROLLER SIMULATION
# ============================================================

def simulate_PI(Kc, tauI):

    """
    Simulate the closed-loop system using a PI controller.

    Kc   = proportional gain
    tauI = integral time constant

    Returns:
        t   = time
        y_r = output response to reference step
        y_d = output response to disturbance step
        u_r = control input response to reference step
        u_d = control input response to disturbance step
    """

    # --------------------------------------------------------
    # PI controller
    #
    #              tauI*s + 1
    # Gc(s) = Kc ----------------
    #                 tauI*s
    # --------------------------------------------------------

    Gc = Kc * control.tf(
        [tauI, 1],
        [tauI, 0]
    )


    # --------------------------------------------------------
    # Closed-loop transfer functions
    # --------------------------------------------------------

    # Reference -> output
    Hyr = (
        Gp * Gv * Gc
        / (1 + Gp * Gv * Gc * Gm)
    )

    # Disturbance -> output
    Hyd = (
        Gd
        / (1 + Gp * Gv * Gc * Gm)
    )

    # Reference -> control input
    Hur = (
        Gc
        / (1 + Gc * Gm * Gp * Gv)
    )

    # Disturbance -> control input
    Hud = (
        -Gc * Gm * Gd
        / (1 + Gc * Gm * Gp * Gv)
    )


    # --------------------------------------------------------
    # Simulation time
    # --------------------------------------------------------

    t = np.linspace(0, 25, 1000)


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # control.step_response() returns:
    #
    #     time, response
    #
    # NOT:
    #
    #     response, time
    # --------------------------------------------------------

    t, y_r = control.step_response(Hyr, t)
    _, y_d = control.step_response(Hyd, t)

    _, u_r = control.step_response(Hur, t)
    _, u_d = control.step_response(Hud, t)


    return t, y_r, y_d, u_r, u_d


# ============================================================
# 4. ZIEGLER-NICHOLS PI PARAMETERS
# ============================================================

# From the example:
#
# Ultimate gain:
# Kcu = 8.1
#
# Ultimate period:
# Pu = 8
#
# Ziegler-Nichols PI:
#
# Kc   = 0.45*Kcu
# tauI = Pu/1.2

Kcu = 8.1
Pu = 8


Kc = 0.45 * Kcu
tauI = Pu / 1.2


# ============================================================
# 5. RUN SIMULATION
# ============================================================

t, y_r, y_d, u_r, u_d = simulate_PI(
    Kc,
    tauI
)


# ============================================================
# 6. PLOT RESULTS
# ============================================================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(12, 6)
)


# ------------------------------------------------------------
# Top left: reference -> output
# ------------------------------------------------------------

axs[0, 0].plot(t, y_r)

axs[0, 0].set_title(
    "Output Response to Step Reference"
)

axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("y")

axs[0, 0].set_ylim(-0.5, 2.2)

axs[0, 0].grid(True)


# ------------------------------------------------------------
# Top right: disturbance -> output
# ------------------------------------------------------------

axs[0, 1].plot(t, y_d)

axs[0, 1].set_title(
    "Output Response to Step Disturbance"
)

axs[0, 1].set_xlabel("Time")
axs[0, 1].set_ylabel("y")

axs[0, 1].set_ylim(-0.5, 2.2)

axs[0, 1].grid(True)


# ------------------------------------------------------------
# Bottom left: reference -> control input
# ------------------------------------------------------------

axs[1, 0].plot(t, u_r)

axs[1, 0].set_title(
    "Control Input Response to Step Reference"
)

axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("u")

axs[1, 0].set_ylim(-1.5, 1.5)

axs[1, 0].grid(True)


# ------------------------------------------------------------
# Bottom right: disturbance -> control input
# ------------------------------------------------------------

axs[1, 1].plot(t, u_d)

axs[1, 1].set_title(
    "Control Input Response to Step Disturbance"
)

axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("u")

axs[1, 1].set_ylim(-1.5, 1.5)

axs[1, 1].grid(True)


# ============================================================
# 7. OVERALL TITLE
# ============================================================

fig.suptitle(
    f"PI Controller Response "
    f"(Kc = {Kc:.3f}, tauI = {tauI:.3f})"
)

plt.tight_layout()

plt.show()