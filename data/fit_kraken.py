#!/usr/bin/env python
import datetime

import matplotlib
import matplotlib.legend
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
import seaborn as sns

import lttb

pd.plotting.register_matplotlib_converters()

from file_parser import parse_file

colors = sns.color_palette("colorblind")

# Use these settings for the PhD thesis
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
    #"pgf.texsystem": "lualatex",
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage{siunitx}",
    ]),
    "savefig.directory": os.path.dirname(os.path.realpath(__file__)),
}
plt.rcParams.update(tex_fonts)
plt.style.use('tableau-colorblind10')
# end of settings


def exponential_decay(t, a, tau, t0, offset):
    # Create a step function that is 0 for time values < t0 and 1 for the rest
    S = [0 if value < t0 else 1 for value in t]

    model = a * np.exp(-(t - t0) / tau * S) + offset
    return model


def fit_exponential_decay(x_data, y_data, initial_theta):
    t = x_data.values
    initial_t0 = 0
    initial_offset = min(y_data.values)
    initial_start = max(y_data.values) - initial_offset
    return curve_fit(
        exponential_decay,
        t,
        y_data.values,
        p0=[initial_start, 1e3, initial_t0, initial_offset],
        bounds=([-np.inf, -np.inf, 0, 0], np.inf),
    )


class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""

    def __init__(self, order_of_mag=0, useOffset=False, useMathText=True):
        super().__init__(useOffset=useOffset, useMathText=useMathText)
        if order_of_mag != 0:
            self.set_powerlimits(
                (
                    order_of_mag,
                    order_of_mag,
                )
            )

    def _set_offset(self, range):
        mean_locs = np.mean(self.locs)

        if range / 2 < np.absolute(mean_locs):
            ave_oom = np.floor(np.log10(mean_locs))
            p10 = 10 ** np.floor(np.log10(range))
            self.offset = np.ceil(np.mean(self.locs) / p10) * p10
        else:
            self.offset = 0


def make_format(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x, y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return "Left: {:<40}    Right: {:<}".format(
            *[
                "({}, {:.6E})".format(
                    matplotlib.dates.num2date(x).strftime("%H:%M:%S"), y
                )
                for x, y in coords
            ]
        )

    return format_coord


def downsample_data(x_data, y_data):
    # This is hacky
    x_is_time = False
    dtype = None
    if pd.api.types.is_datetime64_any_dtype(x_data):
        x_is_time = True
        dtype =  x_data.dtype
        x_data = pd.to_datetime(x_data).astype(np.int64)

    x_data, y_data = lttb.downsample(np.array([x_data, y_data]).T, n_out=1000, validators=[]).T

    if x_is_time:
        x_data = pd.to_datetime(x_data, utc=True)

    return x_data, y_data


def load_data(plot_file):
    print(f"  Parsing: '{plot_file['filename']}'...")
    data = parse_file(**plot_file)

    return data


def crop_data(data, zoom_date=None, crop_secondary=None):
    if zoom_date is not None:
        index_to_drop = data[
            (data.date < zoom_date[0]) | (data.date > zoom_date[1])
        ].index
        data.drop(index_to_drop, inplace=True)

    # y = 1/(0.000858614 + 0.000259555 * np.log(y) + 1.35034*10**-7 * np.log(y)**3)
    print(f"    Begin date: {data.date.iloc[0].tz_convert('Europe/Berlin')}")
    print(
        f"    End date:   {data.date.iloc[-1].tz_convert('Europe/Berlin')} (+{(data.date.iloc[-1]-data.date.iloc[0]).total_seconds()/3600:.1f} h)"
    )


def prepare_axis(ax, axis_settings):
  if axis_settings.get("fixed_order") is not None:
    ax.yaxis.set_major_formatter(FixedOrderFormatter(axis_settings["fixed_order"], useOffset=True))
  else:
    ax.yaxis.get_major_formatter().set_useOffset(False)

  if axis_settings.get("y_scale") == "log":
    ax.set_yscale('log')
  if axis_settings.get("x_scale") == "log":
    ax.set_xscale('log')
  if axis_settings.get("x_scale") == "time":
      ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
      #ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
      ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
  if axis_settings.get("invert_y"):
    ax.invert_yaxis()
  if axis_settings.get("invert_x"):
    ax.invert_xaxis()

  if axis_settings.get("limits_y"):
    ax.set_ylim(*axis_settings.get("limits_y"))

  if axis_settings.get("show_grid", True):
    ax.grid(True, which="minor", ls="-", color='0.85')
    ax.grid(True, which="major", ls="-", color='0.45')
  else:
    ax.grid(False, which="both")

  ax.set_ylabel(axis_settings["y_label"])
  if axis_settings.get("x_label") is not None:
    ax.set_xlabel(axis_settings["x_label"])


def plot_data(ax, data, x_axis, column_settings):
    for column, settings in column_settings.items():
        if column in data:
            data_to_plot = data[[x_axis, column]].dropna()
            if len(data_to_plot) > 1000:
                x_data, y_data = downsample_data(*(data_to_plot[idx] for idx in data_to_plot))
            else:
                x_data, y_data = (data_to_plot[idx] for idx in data_to_plot)
            print(f"  Plotting {len(x_data)} values.")
            ax.plot(
                x_data,
                y_data,
                marker="",
                alpha=0.7,
                **settings
            )


def plot_series(plot):
    # Load the data to be plotted
    plot_files = (
        plot_file for plot_file in plot["files"] if plot_file.get("show", True)
    )
    data = pd.concat((load_data(plot_file)[0] for plot_file in plot_files), sort=True)
    # Drop non-complete rows
    if plot.get("secondary_axis", {}).get("show", True):
        data.dropna(
            subset=[
                list(plot["primary_axis"]["columns_to_plot"])[0],
                list(plot["secondary_axis"]["columns_to_plot"])[0],
            ],
            inplace=True,
        )  # It is ok to drop a few values
    data.reset_index(drop=True, inplace=True)

    # If we have something to plot, proceed
    if not data.empty:
        crop_data(
            data,
            zoom_date=plot.get("zoom"),
            crop_secondary=plot.get("crop_secondary_to_primary"),
        )

        plot_settings = plot["primary_axis"]

        ax1 = plt.subplot(111)
        prepare_axis(ax=ax1, axis_settings=plot_settings["axis_settings"])

        # Reset the time axis to 0 at the unit step
        step_index = data[
            data[list(plot_settings["columns_to_plot"])[0]]
            != data[list(plot_settings["columns_to_plot"])[0]].shift()
        ].index[1]
        data.date = (data.date - data.date[step_index]).dt.total_seconds()

        plot_data(
            ax1,
            data,
            plot_settings["x-axis"],
            plot_settings["columns_to_plot"],
        )
        lines, labels = ax1.get_legend_handles_labels()

        if plot.get("secondary_axis", {}).get("show", True):
            ax2 = ax1.twinx()
            plot_settings2 = plot["secondary_axis"]
            prepare_axis(ax=ax2, axis_settings=plot_settings2["axis_settings"])

            # We need a initial guess of the dead time
            # We assume it is 0
            params, pcov = fit_exponential_decay(
                pd.to_numeric(data.date),
                data[list(plot_settings2["columns_to_plot"])[0]],
                initial_theta=0,
            )
            # Calculate the uncertainty (95%) of the constants
            alpha = 1 - 0.954499736103642  # 95% confidence interval = 100*(1-alpha)
            n = len(data)  # number of data points
            n_params = len(params)  # number of constants
            dof = max(0, n - n_params)  # number of degrees of freedom
            # student-t value for the dof and confidence level
            tval = t.ppf(1.0 - alpha / 2.0, dof)
            sigma_squared = np.diag(pcov)
            output_step = max(data[list(plot_settings["columns_to_plot"])[0]]) - min(
                data[list(plot_settings["columns_to_plot"])[0]]
            )
            system_step = params[0]
            K = system_step / output_step
            T = params[1]
            tau = params[2]
            print(
                f"    Normalized step K: {system_step} K / {output_step} bit ({K} ± {sigma_squared[0]**0.5*tval/output_step}) K/bit"
            )
            print(f"    Decay time T ({T} ± {sigma_squared[1]**0.5*tval}) s")
            print(f"    Dead time τ: ({tau} ± {sigma_squared[2]**0.5*tval}) s")
            print(
                f"    PI parameters (Ziegler–Nichols): Kp={0.9*T/(K*tau)} bit/K, Ki={0.3*T/(K*tau**2)} bit/(Ks)"
            )
            print(
                f"    PI parameters (SIMC): Kp={T/(2*K*tau)} bit/K, Ki={T/(2*K*tau)/np.minimum(T, 8*tau)} bit/(Ks)"
            )
            print(
                f"    PI parameters (Feedback Systems): Kp={(0.15*tau+0.35*T)/(K*tau)} bit/K, Ki={(0.46*tau+0.02*T)/(K*tau**2)} bit/(Ks)"
            )
            k_amigo = 0.15/K + (0.35-(tau*T)/(tau+T)**2)*T/(K*tau)
            print(
                f"    PI parameters (AMIGO): Kp={k_amigo} bit/K, Ki={k_amigo/(0.35*tau+13*tau*T**2/(T**2+12*tau*T+7*tau**2))} bit/(Ks)"
            )
            print(
                f"    PI parameters (APQ): Kp={(0.15*tau+0.35*T)/(K*tau)/6:.2f} bit/K, Ki={(0.46*tau+0.02*T)/(K*tau**2)/4:.2f} bit/(Ks)"
            )
            print(f"    PID setpoint: {system_step+params[3]:.1f} °C")
            data["fit"] = exponential_decay(pd.to_numeric(data.date.values), *params)

            plot_data(
                ax2,
                data,
                plot_settings["x-axis"],
                plot_settings2["columns_to_plot"],
            )
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2

        # ax1.set_ylabel(plot_settings['label'])
        # ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
        # ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax1.set_xlabel(r"Time in \unit{\second}")

        plt.legend(lines, labels, loc="best")

    fig = plt.gcf()
    #  fig.set_size_inches(11.69,8.27)   # A4 in inch
    #  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
    phi = (5**.5-1) / 2  # golden ratio
    fig.set_size_inches(441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi)  # TU thesis
    if plot.get("title") is not None:
        plt.suptitle(plot["title"], fontsize=16)

    plt.tight_layout()
    if plot.get("title") is not None:
        plt.subplots_adjust(top=0.88)

    if plot.get("output_file"):
        print(f"    Saving image to '{plot['output_file']['fname']}'")
        plt.savefig(**plot["output_file"])
    plt.show()


if __name__ == "__main__":
    plots = [
        {
            "title": "K01 Server Room",
            "title": None,
            "show": False,
            #'zoom': ['2022-09-13 04:34:00', '2025-06-26 00:00:00'],  # Only exponential decay
            "zoom": [
                "2022-09-13 03:30:00",
                "2025-09-13 04:30:00",
            ],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": r"DAC outpt in \unit{\bit}",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": r"Temperature in \unit{\celsius}",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature": "Temperature (Labnode)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663079073167.csv",  # Server Room, 10k/10k resistors
                    "show": True,
                    "parser": "timescale_db_2",
                    "options": {
                        "scaling": {
                            "temperature": lambda x: x["temperature"] - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "011 Neon Lab (Front)",
            "title": None,
            "show": False,
            #'zoom': ['2022-09-13 03:30:00', '2025-09-13 04:30:00'],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "DAC outpt in bit",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature": "Temperature (Labnode)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663476153397.csv",  # Neon front, 10k/10k resistors
                    "show": True,
                    "parser": "timescale_db_2",
                    "options": {
                        "scaling": {
                            "temperature": lambda x: x["temperature"] - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "011 Neon Lab (Back)",
            "title": None,
            "show": True,
            "output_file": {
                "fname": "../images/pid_parameter_fit.pgf"
            },
            "zoom": [
                "2022-09-13 03:30:00",
                "2022-09-22 07:20:00",
            ],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "axis_settings": {
                    'x_label': r"Time in \unit{\s}",
                    'y_label': r"DAC outpt in \unit{\bit}",
                    "invert_x": False,
                    "invert_y": False,
                    "x_scale": "lin",
                    "y_scale": "lin",
                },
                "x-axis": "date",
                "label": r"DAC outpt in \unit{\bit}",
                "axis_fixed_order": 0,
                "columns_to_plot": {
                    "output": {
                        "label": "DAC output",
                        "color": colors[4],
                        "linewidth": 2.5,
                    },
                },
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "plot_type": "absolute",
                "axis_settings": {
                    'x_label': r"Time in \unit{\s}",
                    'y_label': r"Temperature in \unit{\celsius}",
                    "invert_x": False,
                    "invert_y": False,
                    "x_scale": "lin",
                    "y_scale": "lin",
                    "show_grid": False,
                },
                "columns_to_plot": {
                    "temperature_labnode" : {
                        "label": "Temperature (LabNode)",
                        "color": colors[0],
                        "linewidth": 0.5,
                    },
                    "fit" : {
                        "label": "Fit",
                        "color": colors[1],
                        "linewidth": 2,
                    },
                },
                "labels": {
                    "temperature_labnode": "Temperature (Labnode)",
                    "temperature_room": "Temperature (Aircon)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663917584771.csv",  # Neon back, 10k/10k resistors
                    "filename": "data-1683380356796.csv",  # Same as above, but with the null filled using locf()
                    "show": True,
                    'parser': 'ltspice_fets',
                    'options': {
                        "columns": {
                            0: "date",
                            1: "output",
                            2: "temperature_room",
                            3: "temperature_labnode",
                        },
                        "scaling": {
                            "date": lambda x: pd.to_datetime(x.date, utc=True),
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                            "temperature_room": lambda x: x["temperature_room"]
                            - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "012 Laser Lab (Front)",
            "title": None,
            "show": False,
            #'zoom': ['2022-09-22 18:30:00', '2022-09-22 20:00:00'],  # Failed attempt
            #'zoom': ['2022-09-22 18:30:00', '2022-09-22 20:00:00'],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "DAC outpt in bit",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature_labnode"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature_labnode": "Temperature (Labnode)",
                    "temperature_room": "Temperature (Room)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663924972026.csv",  # Failed attempt, moved controller
                    "show": False,
                    "parser": "timescale_db_3",
                    "options": {
                        "scaling": {
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                        },
                    },
                },
                {
                    "filename": "data-1663918677781.csv",  # Laser lab front, 10k/10k resistors
                    "show": True,
                    "parser": "timescale_db_3",
                    "options": {
                        "scaling": {
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "012 Laser Lab (Back)",
            "title": None,
            "show": False,
            "zoom": [
                "2022-09-21 02:30:00",
                "2022-09-21 04:20:00",
            ],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "DAC outpt in bit",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature_labnode"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature_labnode": "Temperature (Labnode)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663919213658.csv",  # Laser lab back, 10k/10k resistors
                    "show": True,
                    "parser": "timescale_db_3",
                    "options": {
                        "scaling": {
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "015 ATOMICS Lab (Front)",
            "title": None,
            "show": False,
            "zoom": [
                "2021-09-13 03:30:00",
                "2022-09-21 02:40:00",
            ],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "DAC outpt in bit",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature_labnode"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature_labnode": "Temperature (Labnode)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663737298096.csv",  # ATOMICS front, 10k/10k resistors
                    "show": True,
                    "parser": "timescale_db_3",
                    "options": {
                        "scaling": {
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "015 ATOMICS Lab (Back)",
            "title": None,
            "show": False,
            "zoom": [
                "2021-09-13 03:30:00",
                "2022-09-22 21:10:00",
            ],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "DAC outpt in bit",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature_labnode"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature_labnode": "Temperature (Labnode)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1663913960965.csv",  # ATOMICS back, 10k/10k resistors
                    "show": True,
                    "parser": "timescale_db_3",
                    "options": {
                        "scaling": {
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": "016 Ti:Sa",
            "title": None,
            "show": False,
            #'zoom': ['2021-09-13 03:30:00', '2022-09-22 21:10:00'],  # This range starts reasonably flat
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "DAC outpt in bit",
                "axis_fixed_order": 0,
                "columns_to_plot": [
                    "output",
                ],
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "output": "DAC output",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": True,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["temperature_labnode"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "temperature_labnode": "Temperature (Labnode)",
                    "fit": "Fit",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "data-1664095658934.csv",  # ATOMICS back, 10k/10k resistors
                    "show": True,
                    'parser': 'ltspice_fets',
                    'options': {
                        "columns": {
                            0: "date",
                            1: "output",
                            2: "temperature_room",
                            3: "temperature_labnode",
                        },
                        "scaling": {
                            "temperature_labnode": lambda x: x["temperature_labnode"]
                            - 273.15,
                        },
                    },
                },
            ],
        },
    ]

    plots = (plot for plot in plots if plot.get("show", True))
    for plot in plots:
        print("Ploting {plot!s}".format(plot=plot["title"]))
        plot_series(plot=plot)
