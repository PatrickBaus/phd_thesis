#!/usr/bin/env python
import datetime

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

    # If we have something to plot, proceed
    if not data.empty:
        crop_data(
            data,
            zoom_date=plot.get("zoom"),
            crop_secondary=plot.get("crop_secondary_to_primary"),
        )

        plot_settings = plot["primary_axis"]
        process_data(
            data=data,
            columns=plot_settings["columns_to_plot"],
            plot_type=plot_settings.get("plot_type", "absolute"),
        )

        ax1 = plt.subplot(311)
        plt.tick_params("x", labelbottom=False)
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

        ax = plt.subplot(312, sharex=ax1, sharey=ax1)
        ax.set_ylabel(plot_settings["label"])
        plt.tick_params("x", labelbottom=False)
        plot_data(
            ax,
            data,
            plot_settings["columns_to_plot"][1],
            plot_settings["labels"],
            linewidth=0.5,
        )

        ax = plt.subplot(313, sharex=ax1, sharey=ax1)
        plot_data(
            ax,
            data,
            plot_settings["columns_to_plot"][2],
            plot_settings["labels"],
            linewidth=0.5,
        )
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time (UTC)")

        lines, labels = ax.get_legend_handles_labels()

    fig = plt.gcf()
    #  fig.set_size_inches(11.69,8.27)   # A4 in inch
    #  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
    fig.set_size_inches(
        418.25555 / 72.27 * 0.9, 418.25555 / 72.27 * (5**0.5 - 1) / 2 * 0.9
    )
    if plot.get("title") is not None:
        plt.suptitle(plot["title"], fontsize=16)

    plt.tight_layout()
    if plot.get("title") is not None:
        plt.subplots_adjust(top=0.88)
    plt.show()


if __name__ == "__main__":
    plots = [
        {
            "title": "LM399 Burnin",
            "title": None,
            "show": True,
            #'zoom': ['2021-12-03 12:30:00', '2022-04-01 08:00:00'],
            "zoom": [
                "2022-08-31 06:00:00",
                "2024-09-01 06:00:00",
            ],  # popcorn noise comparison, used in PhD thesis
            "crop_secondary_to_primary": True,
            "primary_axis": {
                "label": "Voltage deviation in V",
                "plot_type": "relative",  # absolute, relative, proportional
                "axis_fixed_order": 0,
                "columns_to_plot": [f"K2002 CH{channel}" for channel in range(1, 11)],
                #'columns_to_plot': [f"K2002 CH{channel}" for channel in range(1,11) if channel not in (1,2,3,5,8,9,10,)],
                #'columns_to_plot': ["K2002 CH1"],
                "columns_to_plot": [
                    "K2002 CH1",
                    "K2002 CH9",
                    "K2002 CH10",
                ],  # popcorn noise comparison, used in PhD thesis
                "filter": None,  # filter_savgol(window_length=101, polyorder=3),
                "labels": {
                    f"K2002 CH{channel}": f"Diode {channel}" for channel in range(1, 11)
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
                    "temp_10k": "Temperature (DUT)",
                    "temp_chamber": "Ambient Temperature (DUT)",
                    "temp_100": "Ambient Temperature (DMM)",
                    "DMM6500": "Ambient Temperature (Room)",
                    "humidity": "Ambient Humidity (DUT)",
                    "current_tec": "TEC Current",
                    "voltage_tec": "TEC Voltage",
                    "setpoint": "Setpoint",
                    "temp_tec": "Ambient Temperature (In loop)",
                    "tmp236": "Reference Module Temperature",
                },
                #        'axis_fixed_order': -6,
                "options": {
                    "invert_axis": False,
                },
            },
            "files": [
                {
                    "filename": "LM399_popcorn_noise_test_2022-08-19_18:00:34+00:00.csv",
                    "show": False,
                    "parser": "scan2000",
                    "options": {
                        "convert_temperature": True,
                        #'remove_outliers': {
                        #  'sigma': 2,
                        # },
                        "scaling": {
                            #'K2002 CH10': lambda x : -x,
                        },
                    },
                },
                {
                    "filename": "LM399_popcorn_noise_test_2022-12-22_16:04:29+00:00.csv",
                    "show": True,
                    "parser": "scan2000",
                    "options": {
                        "convert_temperature": True,
                        #'remove_outliers': {
                        #  'sigma': 2,
                        # },
                        "scaling": {
                            #'K2002 CH10': lambda x : -x,
                        },
                    },
                },
                {
                    "filename": "LM399_popcorn_noise_test_2022-12-29_10:30:13+00:00.csv",
                    "show": True,
                    "parser": "scan2000",
                    "options": {
                        "convert_temperature": True,
                        #'remove_outliers': {
                        #  'sigma': 2,
                        # },
                        "scaling": {
                            #'K2002 CH10': lambda x : -x,
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
