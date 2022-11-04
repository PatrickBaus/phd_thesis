#!/usr/bin/env python
import allantools as at
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import random

from markov_chain import ContinuousTimeMarkovModel

# Use these setting for the PhD thesis
tex_fonts = {
    "text.usetex": True,  # Use LaTeX to write all text
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "text.latex.preamble": "\n".join(
        [  # plots will use this preamble
            r"\usepackage{siunitx}",
        ]
    ),
    "pgf.preamble": "\n".join(
        [  # plots will use this preamble
            r"\usepackage{siunitx}",
        ]
    ),
    "savefig.directory": os.chdir(os.path.dirname(__file__)),
}
plt.rcParams.update(tex_fonts)
plt.style.use("seaborn-colorblind")  # or tableau-colorblind10
# end of settings


def gen_burst_noise(
    number_of_samples: int,
    samplerate: float,
    tau1: float,
    tau0: float = 1,
    std_gaussian_noise: float = 0,
    uniform_noise: float = 0,
    offset: float = 0,
) -> np.ndarray:
    rts_model = ContinuousTimeMarkovModel(
        ["down", "up"],
        [1 / tau0 / samplerate, 1 / tau1 / samplerate],
        np.array([[0.0, 1], [1, 0]]),
    )

    data = rts_model.generate_sequence(number_of_samples, delta_time=1, seed=42)

    if uniform_noise != 0:
        data = data + uniform_noise * (
            np.random.rand(
                data.size,
            )
            - 0.5
        )
    if std_gaussian_noise != 0:
        data = data + np.random.normal(0, std_gaussian_noise, data.size)
    return data + offset


def burst_noise_psd(f, tau1, tau0=1):
    return 4 / ((tau1 + tau0) * ((1 / tau1 + 1 / tau0) ** 2 + f**2 * (4 * np.pi**2)))


def burst_noise_adev(T, tau1, tau0=1, delta_y=1):
    Rxx0 = delta_y**2 * (tau1 * tau0) / (tau1 + tau0) ** 2
    tau_mean = 1 / (1 / tau1 + 1 / tau0)

    avar = (
        Rxx0 * tau_mean**2 / T**2 * (4 * np.exp(-T / tau_mean) - np.exp(-2 * T / tau_mean) + 2 * T / tau_mean - 3)
    )
    return np.sqrt(avar)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # We misuse a python dict, which maintains insertion order, as an ordered set
    plots_to_show = dict.fromkeys(
        [
            "amplitude",
            "psd",
            "adev",
        ]
    )
    plots_to_show = dict.fromkeys(
        [
            "amplitude",
        ]
    )
    plot_direction = "vertical"  # or "horizontal"
    plot_direction = "horizontal"  # or "vertical"

    nperseg = 100
    N = int(20e6) if "adev" in plots_to_show else int(2e6)  # use 20e6 for the adev plot else 2e6
    fs = 2e2

    tau1s = [0.1, 1, 10]
    # tau1s = [1, ]

    # compute amplitude time series
    amplitudes = [
        gen_burst_noise(number_of_samples=N, samplerate=fs, tau1=tau1, tau0=1, offset=i) for i, tau1 in enumerate(tau1s)
    ]

    if "psd" in plots_to_show:
        # compute amplitude PSD
        psds = [signal.welch(noise, fs, nperseg=len(noise) / nperseg, window="hann") for noise in amplitudes]
        psds_theo = [signal.welch(noise, fs, nperseg=len(noise) / nperseg, window="hann") for noise in amplitudes]

    if "adev" in plots_to_show:
        # compute ADEV
        taus = np.logspace(-2, 2, num=20)
        adevs = [at.oadev(noise, data_type="freq", rate=fs, taus=taus)[:2] for noise in amplitudes]

    fig, axs = plt.subplots(
        len(plots_to_show) if plot_direction != "horizontal" else 1,
        len(plots_to_show) if plot_direction == "horizontal" else 1,
        layout="constrained",
    )
    axs = (
        [
            axs,
        ]
        if len(plots_to_show) == 1
        else axs
    )

    # Amplitude plot
    if "amplitude" in plots_to_show:
        number_point_to_plot = 2000
        ax = axs[list(plots_to_show).index("amplitude")]
        for tau1, amplitude in zip(tau1s, amplitudes):
            ax.step(
                np.arange(number_point_to_plot) / fs,
                amplitude[:number_point_to_plot],
                label=f"$\\bar\\tau_1=\\qty{{{tau1}}}{{\\s}}$",
            )  # , color=colors[tau])
        ax.grid(True, which="major", ls="-", color="0.45")
        ax.legend(loc="upper right")
        # ax.set_title(r'Time Series')
        ax.set_xlabel(r"Time in \unit{\second}")
        ax.set_ylabel(r"Amplitude in arb. unit")

    # PSD plot
    if "psd" in plots_to_show:
        ax = plt.subplot(
            len(plots_to_show) if plot_direction != "horizontal" else 1,
            len(plots_to_show) if plot_direction == "horizontal" else 1,
            list(plots_to_show).index("psd") + 1,
        )
        plt.gca().set_prop_cycle(None)  # Reset the color cycle

        for tau1, (freqs, psd) in zip(tau1s, psds):
            (lines,) = ax.loglog(
                freqs,
                [burst_noise_psd(freq, tau1) for freq in freqs],
                "--",
                label=f"$\\bar\\tau_1=\\qty{{{tau1}}}{{\\s}}$",
            )
            ax.loglog(freqs, psd, ".", color=lines.get_color(), markersize=2)

        ax.grid(True, which="minor", ls="-", color="0.85")
        ax.grid(True, which="major", ls="-", color="0.45")
        # ax.set_ylim(5e-2, 5e6)  # Set limits, so that all plots look the same
        ax.legend(loc="lower left")
        # ax.set_title(r'Frequency Power Spectral Density')
        ax.set_xlabel(r"Frequency in $\unit{\Hz}$")
        ax.set_ylabel(r" $S_y(f)$ in $\unit{1 \per \Hz}$")

    # ADEV plot
    if "adev" in plots_to_show:
        ax = plt.subplot(
            len(plots_to_show) if plot_direction != "horizontal" else 1,
            len(plots_to_show) if plot_direction == "horizontal" else 1,
            list(plots_to_show).index("adev") + 1,
        )
        plt.gca().set_prop_cycle(None)  # Reset the color cycle

        for tau1, (taus, adev) in zip(tau1s, adevs):
            (lines,) = ax.loglog(taus, [burst_noise_adev(T=tau, tau1=tau1, tau0=1) for tau in taus], "--")
            ax.loglog(
                taus, adev, "o", markersize=3, label=f"$\\bar\\tau_1=\\qty{{{tau1}}}{{\\s}}$", color=lines.get_color()
            )

        ax.legend(loc="upper right")
        ax.grid(True, which="minor", ls="-", color="0.85")
        ax.grid(True, which="major", ls="-", color="0.45")
        # ax.set_ylim(1e-2, 1e4)  # Set limits, so that all plots look the same
        # ax.set_title(r'Allan Deviation')
        ax.set_xlabel(r"$\tau$ in \unit{\second}")
        ax.set_ylabel(r"ADEV $\sigma_A(\tau)$")

    #    fig.set_size_inches(11.69,8.27)   # A4 in inch
    #    fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
    phi = (5**0.5 - 1) / 2 if plot_direction == "horizontal" else (5**0.5 + 1) / 2  # golden ratio
    # phi = 1
    scale = 0.3 * len(plots_to_show)
    scale = 2 / 3  # scale to 0.9 for (almost) full text width
    fig.set_size_inches(441.01773 / 72.27 * scale, 441.01773 / 72.27 * scale * phi)

    plt.show()
