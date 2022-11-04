#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import allantools as at
import math
import os

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
    # "pgf.texsystem": "lualatex",
    "pgf.preamble": "\n".join(
        [  # plots will use this preamble
            r"\usepackage{siunitx}",
        ]
    ),
    "savefig.directory": os.chdir(os.path.dirname(__file__)),
}
plt.rcParams.update(tex_fonts)
# end of settings

# Select a color style. We do not use plt.style.use(), because the colors need to assigned in a fixed order according
# to the power law plot
color_style = "seaborn-colorblind"
colors = plt.style.library[color_style]["axes.prop_cycle"].by_key()["color"]


def main():
    np.random.seed(42)
    nr = 2**14  # number of datapoints in time-series
    adev0 = 1.0

    tau0 = 1.0  # sample interval
    sample_rate = 1.0 / tau0
    D = (
        1 / tau0 * 2 ** (0.5)
    )  # normalized to the sample rate, similar to qd to make sure all ADEVs start at the same y-value
    # Generate noise: white (beta=-2), flicker (beta=-3), random-walk (beta=-4)
    betas = [-2, -3, -4]
    betas = [
        -2,
    ]
    labels = {-2: "White noise", -3: "Flicker noise", -4: "Random walk"}  # betas  # mu = -1  # mu = 0  # 0 mu = 1
    beta_colors = {beta: colors[-2 - beta] for beta in betas}
    # We misuse a python dict, which maintains insertion order, as an ordered set
    plots_to_show = dict.fromkeys(["amplitude", "psd", "adev"])
    plots_to_show = dict.fromkeys(["adev"])
    plot_direction = "vertical"  # or "horizontal"
    plot_direction = "horizontal"  # or "vertical"

    # discrete variance for noiseGen()
    # We normalize all adevs to adev0 at tau0
    # This is done according to "Discrete simulation of power law noise" eq. 7
    # The coefficients (the first term in brackets) like (2*np.log(2) comes from
    # "Characterization of Frequency Stability" Appendix II
    normalization_coeffients = {
        -2: 0.5,
        -3: 2 * np.log(2),
        -4: 2 * np.pi**2 / 3,
    }
    qd = {
        beta: adev0**2
        / (normalization_coeffients[beta] * 2 * (2 * np.pi) ** beta * tau0 ** (beta + 1) * (2 * np.pi) ** 2)
        for beta in betas
    }

    colored_noise = [at.Noise(nr, qd[beta], beta) for beta in betas]
    for noise in colored_noise:
        noise.generateNoise()

    # compute amplitude time series
    amplitudes = [at.phase2frequency(noise.time_series, sample_rate) for noise in colored_noise]
    drift_amplitude = np.arange(nr - 1) * D + 0  # linear drift

    if "psd" in plots_to_show:
        # compute amplitude PSD
        psds = [at.noise.scipy_psd(amplitude, f_sample=sample_rate, nr_segments=4) for amplitude in amplitudes]

        # compute amplitude PSD prefactor h_a
        has = [noise.frequency_psd_from_qd(tau0) for noise in colored_noise]

    if "adev" in plots_to_show:
        # compute ADEV
        adevs = [at.oadev(noise.time_series, rate=sample_rate, taus="decade")[:2] for noise in colored_noise]
        drift_taus, drift_adev, *_ = at.oadev(drift_amplitude, data_type="freq", rate=sample_rate, taus="decade")

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
        ax = axs[list(plots_to_show).index("amplitude")]
        for beta, amplitude in zip(betas, amplitudes):
            ax.plot(np.arange(nr - 1) * tau0, amplitude, label=labels[beta], color=beta_colors[beta])
        # ax.plot(np.arange(nr-1)*tau0, drift_amplitude, label="Linear drift")
        ax.grid(True, which="major", ls="-", color="0.45")
        ax.set_ylim(-6.5, 6.5)  # Set limits, so that all plots look the same
        ax.legend(loc="upper left")
        # ax.set_title(r'Time Series')
        ax.set_xlabel(r"Time in $\unit{\second}$")
        ax.set_ylabel(r"Ampl. in arb. unit")

    # PSD plot
    if "psd" in plots_to_show:
        ax = plt.subplot(
            len(plots_to_show) if plot_direction != "horizontal" else 1,
            len(plots_to_show) if plot_direction == "horizontal" else 1,
            list(plots_to_show).index("psd") + 1,
        )
        plt.gca().set_prop_cycle(None)  # Reset the color cycle

        for beta, (freqs, psd), ha in zip(betas, psds, has):
            (lines,) = ax.loglog(
                freqs[1:],
                [ha * pow(freq, beta + 2) for freq in freqs[1:]],
                "--",
                label=f"$h_{{{beta+2}}}f^{{{beta+2}}}$",
                color=beta_colors[beta],
            )
            ax.loglog(freqs, psd, ".", color=lines.get_color(), markersize=2)

        ax.grid(True, which="minor", ls="-", color="0.85")
        ax.grid(True, which="major", ls="-", color="0.45")
        ax.set_ylim(5e-2, 5e6)  # Set limits, so that all plots look the same
        ax.legend(loc="upper right")
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

        for noise, (taus, adev) in zip(colored_noise, adevs):
            (lines,) = ax.loglog(
                taus,
                [noise.adev_from_qd(tau0, tau) * tau ** ((-3 - noise.b) / 2) for tau in taus],
                "--",
                label=f"$\\propto\\sqrt{{h_{{{noise.b+2}}}}}\\tau^{{{(-3-noise.b)/2:+}}}$",
                color=beta_colors[noise.b],
            )
            ax.loglog(taus, adev, "o", color=lines.get_color(), markersize=3)

        # lines, = ax.loglog(drift_taus, [math.sqrt(0.5)*D*(tau/tau0) for tau in drift_taus],
        #        '--', label=r'$\propto D\tau^{+1}$')
        # ax.loglog(drift_taus, drift_adev, 'o', color=lines.get_color(), markersize=3)

        ax.legend(loc="best")
        ax.grid(True, which="minor", ls="-", color="0.85")
        ax.grid(True, which="major", ls="-", color="0.45")
        ax.set_ylim(1e-2, 5e4)  # Set limits, so that all plots look the same
        # ax.set_title(r'Allan Deviation')
        ax.set_xlabel(r"$\tau$ in \unit{\second}")
        ax.set_ylabel(r"ADEV $\sigma_A(\tau)$")

    #  fig.set_size_inches(11.69,8.27)   # A4 in inch
    #  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
    phi = (5**0.5 - 1) / 2 if plot_direction == "horizontal" else (5**0.5 + 1) / 2  # golden ratio
    phi = 0.75
    scale = 0.3 * len(plots_to_show)
    scale = 0.4  # scale to 0.9 for (almost) full text width, use 0.4 for publication
    fig.set_size_inches(441.01773 / 72.27 * scale, 441.01773 / 72.27 * scale * phi)

    plt.show()


if __name__ == "__main__":
    main()
