#!/usr/bin/env python
import datetime
import os

import matplotlib
import matplotlib.legend
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from si_prefix import si_format

pd.plotting.register_matplotlib_converters()

from file_parser import parse_file

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
plt.style.use("tableau-colorblind10")
# end of settings


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


def filter_savgol(window_length, polyorder):
    def filter(data):
        if len(data) <= window_length:
            return None

        return signal.savgol_filter(data, window_length, polyorder)

    return filter


def filter_butterworth(window_length=0.00005):
    from scipy.signal import butter, filtfilt

    b, a = butter(3, window_length)

    def filter(data):
        return filtfilt(b, a, data)

    return filter


def filter_rolling(window_length):
    def filter(data):
        if len(data) <= window_length:
            return None

        return data.rolling(window=window_length).mean()

    return filter


def process_data(data, columns, plot_type):
    if plot_type == "relative":
        data[columns] = data[columns] - data[columns].mean().tolist()
    elif plot_type == "proportional":
        data[columns] = data[columns] / data[columns].iloc[:30].mean().tolist() - 1


def prepare_axis(ax, label, color_map=None, fixed_order=None):
    if fixed_order is not None:
        ax.yaxis.set_major_formatter(FixedOrderFormatter(fixed_order, useOffset=True))
    else:
        ax.yaxis.get_major_formatter().set_useOffset(False)

    if color_map is not None:
        ax.set_prop_cycle("color", color_map)

    ax.set_ylabel(label)


def plot_data(ax, data, column, labels, linewidth):
    ax.plot(
        data.date,
        data[column],
        marker="",
        label=labels[column],
        alpha=0.7,
        linewidth=linewidth,
    )


def plot_series(plot):
    # Load the data to be plotted
    plot_files = (
        plot_file for plot_file in plot["files"] if plot_file.get("show", True)
    )
    data = pd.concat((load_data(plot_file)[0] for plot_file in plot_files), sort=True)
    data.reset_index(drop=True, inplace=True)

    # If we have something to plot, proceed
    if not data.empty:
        crop_data(
            data,
            zoom_date=plot.get("zoom"),
            crop_secondary=plot.get("crop_secondary_to_primary"),
        )

        plot_settings = plot["primary_axis"]
        print(f"    Mean:   {np.mean(data[plot_settings['columns_to_plot'][0]])} V")
        process_data(
            data=data,
            columns=plot_settings["columns_to_plot"],
            plot_type=plot_settings.get("plot_type", "absolute"),
        )

        ax1 = plt.subplot(111)
        prepare_axis(
            ax=ax1,
            fixed_order=plot_settings["axis_fixed_order"],
            label=plot_settings["label"],
            color_map=plt.cm.tab10.colors,
        )

        plot_data(
            ax1,
            data,
            plot_settings["columns_to_plot"][0],
            plot_settings["labels"],
            linewidth=0.5,
        )
        lines, labels = ax1.get_legend_handles_labels()

        if plot.get("secondary_axis", {}).get("show", True):
            ax2 = ax1.twinx()
            ax2.format_coord = make_format(ax2, ax1)
            plot_settings2 = plot["secondary_axis"]
            prepare_axis(
                ax=ax2,
                fixed_order=plot_settings2.get("axis_fixed_order"),
                label=plot_settings2["label"],
                color_map=[
                    "firebrick",
                    "salmon",
                ],
            )
            plot_data(
                ax2,
                data,
                plot_settings2["columns_to_plot"][0],
                plot_settings2["labels"],
                linewidth=0.5,
            )
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2

        # ax1.set_ylabel(plot_settings['label'])
        # ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax1.set_xlabel("Time (UTC)")

        plt.legend(lines, labels, loc="upper left")

    fig = plt.gcf()
    #  fig.set_size_inches(11.69,8.27)   # A4 in inch
    #  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
    # Latex uses 72.27 pts/inch and the TUD design has a width of 418.26 pts. It can be obtained by
    # using \printpicturesize
    # (5**.5 - 1) / 2) is the golden ratio
    fig.set_size_inches(
        418.25555 / 72.27 * 0.9, 418.25555 / 72.27 * (5**0.5 - 1) / 2 * 0.9
    )
    if plot.get("title") is not None:
        plt.suptitle(plot["title"], fontsize=16)

    plt.tight_layout()
    if plot.get("title") is not None:
        plt.subplots_adjust(top=0.88)

    if plot.get("output_file"):
        print(f"  Saving image to '{plot['output_file']['fname']}'")
        plt.savefig(**plot["output_file"])

    plt.show()


if __name__ == "__main__":
    plots = [
        {
            "title": None,
            "show": True,
            #'zoom': ['2017-06-17 21:30:00', '2017-06-18 00:00:00'], # Serial no. ?, used in PhD thesis
            "zoom": [
                "2017-06-25 00:00:00",
                "2017-06-26 00:00:00",
            ],  # Serial no. ?, used in PhD thesis
            "output_file": {
                "fname": "../images/LM399_vs_34470A.pgf"
            },
            "crop_secondary_to_primary": False,
            "primary_axis": {
                "label": "Voltage deviation in V",
                "plot_type": "relative",  # absolute, relative, proportional
                "axis_fixed_order": -6,
                "columns_to_plot": [
                    "value",
                ],  # popcorn noise comparison, used in PhD thesis
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "value": "DUT vs KS34470A",
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
                    "temperature": "Ambient Temperature",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    # Use with ilter_savgol(window_length=101, polyorder=3), to the find the noise
                    "filename": "LM399_36h.csv",  # LM399 PCB v2.0, 10V,
                    "show": False,
                    "parser": "34470A",
                    "options": {},
                },
                {
                    # Use with ilter_savgol(window_length=101, polyorder=3), to the find the noise
                    "filename": "LM399_84h_noAC.csv",  # LM399 PCB v2.0, 10V,
                    "filename": "LM399_84h_case.csv",  # LM399 PCB v2.0, 10V,
                    "show": True,
                    "parser": "34470A",
                    "options": {},
                },
                {
                    # Use with ilter_savgol(window_length=101, polyorder=3), to the find the noise
                    "filename": "sensorData_2017-06-23 16-42-00_2017-06-27 04-42-00.csv",  # LM399 PCB v2.0, 10V,
                    "show": True,
                    "parser": "smi",
                    "options": {
                        "sensor_id": 8,
                        "label": "temperature",
                        "scaling": {
                            "temperature": lambda x: x["temperature"] - 273.15,
                        },
                    },
                },
            ],
        },
        {
            "title": None,
            "show": True,
            "zoom": [
                "2020-05-26 06:45:00",
                "2020-05-26 07:00:00",
            ],  # Serial no. 15, popcorn noise, used in thesis
            "output_file": {
                "fname": "../images/refurb_lm399_popcorn_noise.pgf"
            },
            "crop_secondary_to_primary": True,
            "primary_axis": {
                "label": "Voltage deviation in V",
                "plot_type": "relative",  # absolute, relative, proportional
                "axis_fixed_order": -6,
                "columns_to_plot": [
                    "value",
                ],  # popcorn noise comparison, used in PhD thesis
                "filter": None,
                # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    "value": "DUT vs KS34470A",
                },
                "options": {
                    "show_filtered_only": False,
                },
            },
            "secondary_axis": {
                "show": False,
                "label": "Temperature in °C",
                "plot_type": "absolute",
                "unit": "°C",
                "columns_to_plot": ["humidity"],
                #        'filter': filter_savgol(window_length=151, polyorder=3),
                "labels": {
                    "value": "DUT vs KS34470A",
                    "tmp236": "Reference Module Temperature",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    # Use with ilter_savgol(window_length=101, polyorder=3), to the find the noise
                    "filename": "DgDrive_reference_3-2-1_noise_2020-05-25_16:45:55+00:00.csv",  # Zener #15 vs. #1
                    "show": True,
                    "parser": "LM399_logger_v2",
                    "options": {},
                },
            ],
        },
    ]

    plots = (plot for plot in plots if plot.get("show", True))
    for plot in plots:
        print("Ploting {plot!s}".format(plot=plot["title"]))
        plot_series(plot=plot)
