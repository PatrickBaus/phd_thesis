#!/usr/bin/env python
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Use these setting for the PhD thesis
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }
)
plt.style.use("tableau-colorblind10")
# end of settings


@dataclass
class model:
    """
    The process model paramters
    """

    k: float
    tau: float
    theta: float


def process(y, t, u, k, tau):
    """
    u: float
        unit step height
    k: float
        process gain
    tau: float
        time constant of the process
    """
    dydt = -y / tau + k / tau * u
    return dydt


def calc_response(t, m, u):
    # Number of steps. Do one more step, because we need to do the unit step `u` in the first step. This will later
    # be stripped
    ns = len(t) - 1 + 1

    delta_t = t[1] - t[0]

    op = np.full(ns + 1, u)  # controller output
    pv = np.zeros(ns + 1)  # process variable

    # step input
    op[0] = 0

    # Simulate time delay
    ndelay = int(np.ceil(m.theta / delta_t))

    # Create the pv by iterating over the time steps
    for i in range(ns):
        # time delay
        iop = max(0, i - ndelay)
        # Integrate the differential equation
        y = odeint(process, pv[i], [0, delta_t], args=(op[iop], m.k, m.tau))
        pv[i + 1] = y[-1]
    return (
        pv[1:],
        op[1:],
    )  # strip off the first value, which was used for the input step


ns = 100
# Create time range
t = np.linspace(0, ns / 10.0, ns + 1)

# Set the model parameters
model.k = 1
model.tau = 2.0
model.theta = 4.0
step_size = 1.0

pv, _ = calc_response(t, model, u=step_size)
pv2 = [
    model.k * (1.0 - np.exp(-(t[i] - model.theta) / model.tau)) for i in range(len(t))
]

ax = plt.subplot(111)
plt.plot(
    t,
    pv2,
    linewidth=2,
    label=r"$1-e^{-\frac{t-\theta}{\tau} }$",
    alpha=0.7,
    linestyle="dashed",
)
plt.plot(
    [t[0], model.theta, model.theta + 0.0001, t[-1]],
    [0, 0, 1, 1],
    linewidth=2,
    label=r"$H(t- \theta)$",
    alpha=0.7,
    linestyle="dotted",
)
next(ax._get_lines.prop_cycler)  # skip the grey colour
plt.plot(t, pv, linewidth=3, label=r"$y(t)$")
plt.legend(loc="best")
plt.ylabel("Process Output")
plt.ylim([-1, 1.5])

plt.xlim([0, 10])

plt.xlabel("Time")

fig = plt.gcf()
#  fig.set_size_inches(11.69,8.27)   # A4 in inch
#  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
fig.set_size_inches(
    418.25555 / 72.27 * 0.9, 418.25555 / 72.27 * (5**0.5 - 1) / 2 * 0.9
)  # TU thesis
plt.tight_layout()

plt.show()
