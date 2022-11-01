#!/usr/bin/env python
import allantools as at
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import signal
import random

from markov_chain import ContinuousTimeMarkovModel, generate_traces

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
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "text.latex.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage{siunitx}",
    ]),
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage{siunitx}",
    ]),
    "savefig.directory": os.chdir(os.path.dirname(__file__)),
}
plt.rcParams.update(tex_fonts)
plt.style.use('seaborn-colorblind')  # or tableau-colorblind10
# end of settings


def gen_burst_noise(number_of_samples: int, samplerate: float, tau1: float, tau0: float = 1,
                    std_gaussian_noise: float = 0, uniform_noise: float = 0
                    ) -> np.ndarray:
    rts_model = ContinuousTimeMarkovModel(['down', 'up'], [1 / tau0 / samplerate,
                                                           1 / tau1 / samplerate], np.array([[0., 1], [1, 0]]),)

    data = rts_model.generate_sequence(number_of_samples, delta_time=1, seed=42)

    if uniform_noise != 0:
        data = data + uniform_noise * (np.random.rand(data.size, ) - .5)
    if std_gaussian_noise != 0:
        data = data + np.random.normal(0, std_gaussian_noise, data.size)
    return data


def burst_noise_psd(f, tau1, tau0=1):
    return 4 / ((tau1 + tau0) * ((1/tau1 + 1/tau0)**2 + f**2 * (4*np.pi**2)) )


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    #fig, axs = plt.subplots(2, 1, layout="constrained")

    nperseg = 100
    N = int(2e6)
    fs = 2e2

    # We misuse a python dict, which maintains insertion order, as an ordered set
    plots_to_show=dict.fromkeys(["amplitude", "psd", "adev"])
    plots_to_show=dict.fromkeys(["psd", "adev"])
    plot_direction = "vertical"  # or "horizontal"
    #plot_direction = "horizontal"  # or "vertical"

    tau1s = [1, 10, 100]
    #tau1s = [1, 10,]

    # compute amplitude time series
    amplitudes = [gen_burst_noise(N, fs, tau1, 1) for tau1 in tau1s]

    if "psd" in plots_to_show:
        # compute amplitude PSD
        psds = [signal.welch(noise, fs, nperseg=len(noise)/nperseg, window='hann') for noise in amplitudes]
        psds_theo = [signal.welch(noise, fs, nperseg=len(noise)/nperseg, window='hann') for noise in amplitudes]

    if "adev" in plots_to_show:
        # compute ADEV
        taus = np.logspace(-2, 2, num=40)
        adevs = [at.oadev(noise, data_type="freq", rate=fs, taus=taus)[:2] for noise in amplitudes]

    fig, axs = plt.subplots(
        len(plots_to_show) if plot_direction!="horizontal" else 1,
        len(plots_to_show) if plot_direction=="horizontal" else 1,
        layout="constrained"
    )
    axs = [axs, ] if len(plots_to_show) == 1 else axs

    # Amplitude plot
    if "amplitude" in plots_to_show:
        ax = axs[list(plots_to_show).index("amplitude")]
        for tau1, amplitude in zip(tau1s, amplitudes):
            ax.step(np.arange(2000)/fs, amplitude[:2000], label=f'$\\tau_1={{{tau1}}}$')#, color=colors[tau])
        ax.grid(True, which="major", ls="-", color='0.45')
        ax.legend(loc='upper left')
        #ax.set_title(r'Time Series')
        ax.set_xlabel(r'Time in \unit{\second}')
        ax.set_ylabel(r'Amplitude in arb. unit')

    # PSD plot
    if "psd" in plots_to_show:
        ax = plt.subplot(
            len(plots_to_show) if plot_direction!="horizontal" else 1,
            len(plots_to_show) if plot_direction=="horizontal" else 1,
            list(plots_to_show).index("psd")+1
        )
        plt.gca().set_prop_cycle(None)  # Reset the color cycle

        for tau1, (freqs, psd) in zip(tau1s, psds):
            lines, = ax.loglog(freqs, [burst_noise_psd(freq, tau1) for freq in freqs], '--', label=f'$\\tau_1={{{tau1}}}$')
            ax.loglog(freqs, psd, '.', color=lines.get_color(), markersize=2)

        ax.grid(True, which="minor", ls="-", color='0.85')
        ax.grid(True, which="major", ls="-", color='0.45')
        #ax.set_ylim(5e-2, 5e6)  # Set limits, so that all plots look the same
        ax.legend(loc='upper right')
        #ax.set_title(r'Frequency Power Spectral Density')
        ax.set_xlabel(r'Frequency in $\unit{\Hz}$')
        ax.set_ylabel(r' $S_y(f)$ in $\unit{1 \per \Hz}$')

    # ADEV plot
    if "adev" in plots_to_show:
        ax = plt.subplot(
            len(plots_to_show) if plot_direction!="horizontal" else 1,
            len(plots_to_show) if plot_direction=="horizontal" else 1,
            list(plots_to_show).index("adev")+1
        )
        plt.gca().set_prop_cycle(None)  # Reset the color cycle

        for tau1, (taus, adev) in zip(tau1s, adevs):
            lines, = ax.loglog(taus, adev, 'o', markersize=3, label=f'$\\tau_1={{{tau1}}}$')
            #lines, = ax.loglog(taus, [noise.adev_from_qd(tau0, tau)*tau**((-3-noise.b)/2) for tau in taus],
            #                '--', label=f'$\propto\sqrt{{h_{{{noise.b+2}}}}}\\tau^{{{(-3-noise.b)/2:+}}}$',
            #                color=beta_colors[noise.b])
            #ax.loglog(taus, adev, 'o', color=lines.get_color(), markersize=3)

        #lines, = ax.loglog(drift_taus, [math.sqrt(0.5)*D*(tau/tau0) for tau in drift_taus],
        #        '--', label=r'$\propto D\tau^{+1}$')
        #ax.loglog(drift_taus, drift_adev, 'o', color=lines.get_color(), markersize=3)

        ax.legend(loc='best')
        ax.grid(True, which="minor", ls="-", color='0.85')
        ax.grid(True, which="major", ls="-", color='0.45')
        #ax.set_ylim(1e-2, 1e4)  # Set limits, so that all plots look the same
        #ax.set_title(r'Allan Deviation')
        ax.set_xlabel(r'$\tau$ in \unit{\second}')
        ax.set_ylabel(r'ADEV $\sigma(\tau)$')

    #noise = gen_burst_noise(N, fs, 100, 1)
    #print(noise[:10])
    #axs[1].plot(noise, ".", markersize=0.5)

    #f, Pxx_den = signal.welch(noise, fs, nperseg=len(noise)/nperseg, window='hann')
    #axs[0].loglog(f, Pxx_den, ".")

    #N = 50

    #freqs = np.logspace(-3, 0, num=N)
    #for tau1 in [1, 10, 100]:
    #for tau1 in [1,]:
    #    axs[0].loglog(freqs, burst_noise_psd(freqs, tau1, 1), marker="", label=f"$\\tau_1={tau1}$", alpha=0.7, linewidth=1)

    #axs[0].grid(True, which="minor", ls="-", color='0.85')
    #axs[0].grid(True, which="major", ls="-", color='0.45')
    #axs[0].set_ylabel(r"Power spectral density in $\unit{\V^2 \per \Hz}$")
    #axs[0].set_xlabel(r"Frequency in \unit{\Hz}")

    #lines, labels = axs[0].get_legend_handles_labels()
    #axs[0].legend(lines, labels, loc='best')

    #plt.ylim([ymin, ymax])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.title(title)


#    fig.set_size_inches(11.69,8.27)   # A4 in inch
#    fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
    phi = (5**.5-1) / 2  # golden ratio
    fig.set_size_inches(441.01773 / 72.27 * 0.9, 441.01773 / 72.27 * 0.9 * phi)

    plt.show()
