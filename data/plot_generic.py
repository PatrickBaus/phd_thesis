#!/usr/bin/env python
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.legend
import numpy as np
import os
import pandas as pd
from matplotlib.ticker import ScalarFormatter
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

class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of 
    magnitude"""
    def __init__(self, order_of_mag=0, useOffset=False, useMathText=True):
        super().__init__(useOffset=useOffset, useMathText=useMathText)
        if order_of_mag != 0:
            self.set_powerlimits((order_of_mag,order_of_mag,))

    def _set_offset(self, range):
        mean_locs = np.mean(self.locs)

        if range / 2 < np.absolute(mean_locs):
            ave_oom = np.floor(np.log10(mean_locs))
            p10 = 10 ** np.floor(np.log10(range))
            self.offset = (np.ceil(np.mean(self.locs) / p10) * p10)
        else:
            self.offset = 0


def make_format(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.3E}, {:.3E})'.format(x, y) for x,y in coords]))
    return format_coord


def load_data(plot_file):
    print(f"  Parsing: '{plot_file['filename']}'...")
    data = parse_file(**plot_file)

    return data

def crop_data(data, crop_index="date", crop=None):
    #data.sort_values(by=crop_index, inplace=True)
    if crop is not None:
        index_to_drop = data[(data[crop_index] < crop[0]) | (data[crop_index] > crop[1])].index if len(crop) > 1 else data[data[crop_index] < crop[0]].index
        data.drop(index_to_drop , inplace=True)

    #y = 1/(0.000858614 + 0.000259555 * np.log(y) + 1.35034*10**-7 * np.log(y)**3)
    #print(f"    Begin date: {data.date.iloc[0].tz_convert('Europe/Berlin')}")
    #print(f"    End date:   {data.date.iloc[-1].tz_convert('Europe/Berlin')} (+{(data.date.iloc[-1]-data.date.iloc[0]).total_seconds()/3600:.1f} h)")

def filter_savgol(window_length, polyorder):
    def filter(data):
        if len(data) <= window_length:
            return None

        return signal.savgol_filter(data, window_length, polyorder)

    return filter

def filter_butterworth(window_length=0.00005):
    from scipy.signal import filtfilt, butter
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

def convertResistanceToTemperature(values):
    # Constants for Amphenol DC95 (Material Type 10kY)
    a = 3.3540153*10**-3
    b = 2.7867185*10**-4
    c = 4.0006637*10**-6
    d = 1.5575628*10**-7
    rt25 = 10*10**3

    return 1 / (a + b * np.log(values / rt25) + c * np.log(values / rt25)**2 + d * np.log(values / rt25)**3) - 273.15

def process_data(data, columns, plot_type):
    if plot_type=='relative':
        data[columns] = data[columns] - data[columns].mean().tolist()
    elif plot_type=='proportional':
        data[columns] = data[columns] / data[columns].iloc[:30].mean().tolist() - 1

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
  if axis_settings.get("x_scale") == "timedelta":
      ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(0,48,3)))
      #ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H"))
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
            if len(data) > 1000:
                x_data, y_data = downsample_data(data[x_axis], data[column])
            else:
                x_data, y_data = data[x_axis], data[column]
            print(f"  Plotting {len(x_data)} values.")
            #x_data, y_data = data[x_axis], data[column]
            ax.plot(
                x_data,
                y_data,
                marker="",
                alpha=0.7,
                **settings
            )

def plot_series(plot):
  # Load the data to be plotted
  plot_files = (plot_file for plot_file in plot['files'] if plot_file.get('show', True))
  data = pd.concat(
    (load_data(plot_file)[0] for plot_file in plot_files),
    sort=True
  )
  data.reset_index(inplace=True)

  # If we have something to plot, proceed
  if not data.empty:
    crop_data(data, **plot.get('crop', {}))

    plot_settings = plot['primary_axis']
    process_data(data=data, columns=plot_settings['columns_to_plot'], plot_type=plot_settings.get('plot_type','absolute'))

    ax1 = plt.subplot(111)
    #plt.tick_params('x', labelbottom=False)
    prepare_axis(ax=ax1, axis_settings=plot_settings["axis_settings"])

    plot_data(ax1, data,  plot_settings["x-axis"], plot_settings['columns_to_plot'])

    lines, labels = ax1.get_legend_handles_labels()

    plot_settings = plot.get('secondary_axis', {})
    if plot_settings.get("show", False):
      ax2 = ax1.twinx()
      prepare_axis(
        ax=ax2,
        axis_settings=plot_settings["axis_settings"],
      )
      ax2.format_coord = make_format(ax2, ax1)
      plot_data(ax2, data, plot_settings["x-axis"], plot_settings['columns_to_plot'])

      lines2, labels2 = ax2.get_legend_handles_labels()
      lines += lines2
      labels += labels2
    if labels:
      plt.legend(lines, labels, loc=plot.get("legend_position", "upper left"))

  fig = plt.gcf()
#  fig.set_size_inches(11.69,8.27)   # A4 in inch
#  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
  if plot.get("plot_size"):
      fig.set_size_inches(*plot["plot_size"])
  else:
      phi = (5**.5-1) / 2  # golden ratio
      fig.set_size_inches(441.01773 / 72.27 * 0.9, 441.01773 / 72.27 * 0.9 * phi)
  if plot.get('title') is not None:
    plt.suptitle(plot['title'], fontsize=16)

  plt.tight_layout()
  if plot.get('title') is not None:
    plt.subplots_adjust(top=0.88)
  if plot.get("output_file"):
    print(f"  Saving image to '{plot['output_file']['fname']}'")
    plt.savefig(**plot["output_file"])
  plt.show()


phi = (5**.5-1) / 2  # golden ratio


if __name__ == "__main__":
  plots = [
    {
      'title': 'DMM comparison vs. Fluke 5440B',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/dmm_comparison_fluke5440B.pgf"
      },
      #'crop': [0,31e-3],
      "legend_position": "lower center",
      'crop_secondary_to_primary': True,
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Voltage deviation in \unit{\V}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -6,
          "x_scale": "time",
          "y_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "KS34470A": {
                "label": r"Keysight 34470A",
                "color": colors[5],
            },
            "HP3458A": {
                "label": r"Keysight 3458A",
                "color": colors[0],
            },
            "K2002": {
                "label": r"Keithley 2002",
                "color": colors[2],
            },
            "DMM6500": {
                "label": r"Keithley DMM6500",
                "color": colors[4],
            },
        },
      },
      'secondary_axis': {
        'show': True,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Ambient temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
          "limits_y": [22.4,23.4],
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Ambient temperature",
                "color": colors[3],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "data-1664633967407.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "KS34470A",
                2: "HP3458A",
                3: "K2002",
                4: "DMM6500",
                5: "temperature",
            },
            'scaling': {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
                "KS34470A": lambda data: data["KS34470A"] - data["KS34470A"].mean() + 2* 10**-5,
                "HP3458A": lambda data: data["HP3458A"] - data["HP3458A"].mean() + 10**-5,
                "K2002": lambda data: data["K2002"] - data["K2002"].mean(),
                "DMM6500": lambda data: data["DMM6500"] - data["DMM6500"].mean() - 10**-5,
            },
          },
        },
      ],
    },
    {
      'title': r'SMC11 stability over \qty{24}{\hour}',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/stability_smc11.pgf"
      },
      'crop': {
        "crop_index": "date",
        "crop": ['2019-12-01 00:00:00', '2019-12-02 00:00:00'],
        "crop": [0, 24],
      },
      "legend_position": "upper right",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time in \unit{\hour}",
          'y_label': r"Current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -6,
          "x_scale": "timedelta",
          "y_scale": "lin",
          "limits_y": [-3e-6,3e-6],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "smc11": {
                "label": r"SMC11",
                "color": colors[0],
            },
            "dgDrive": {
                "label": r"DgDrive-500-LN",
                "color": colors[5],
            },
        },
      },
      'secondary_axis': {
        'show': False,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Ambient temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Ambient temperature",
                "color": colors[0],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "SMC11_50mA+10R_60h_2.csv",
          'show': True,
          'parser': '34470A',
          'options': {
            "value_name": "smc11",
            'scaling': {
                "date": lambda data: (data["date"] - pd.Timestamp("2019-12-01 00:00:00", tz="UTC")).dt.total_seconds() / 3600,
                "smc11": lambda data: (data["smc11"] - data["smc11"].mean())/10,
            },
          },
        },
        {
          'filename': "DgDrive-1-2-1_50mA_FilmCap+RefNo3_10R_34470A_60h_8.csv",
          'show': True,
          'parser': '34470A',
          'options': {
            "value_name": "dgDrive",
            'scaling': {
                "date": lambda data: (data["date"] - pd.Timestamp("2019-11-23 12:00:00", tz="UTC")).dt.total_seconds() / 3600,
                "dgDrive": lambda data: (data["dgDrive"] - data["dgDrive"][(data["date"]>=0) & (data["date"]<=24)].mean())/10 -2e-6,
            },
          },
        },
      ],
    },
    {
      'title': r'DgDrive stability over \qty{24}{\hour}',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/stability_dgDrive.pgf"
      },
      'crop': {
        "crop_index": "date",
      #  "crop": ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
        "crop": [0, 24],
      },
      #"legend_position": "lower center",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time in \unit{\hour}",
          'y_label': r"Current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          "x_scale": "timedelta",
          "y_scale": "lin",
          #"limits_y": [-3e-6,3e-6],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "value": {
                "label": r"DgDrive-500-LN",
                "color": colors[5],
            },
        },
      },
      'secondary_axis': {
        'show': False,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Ambient temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
          #"limits_y": [22.4,23.4],
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Ambient temperature",
                "color": colors[0],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "DgDrive-1-2-1_50mA_FilmCap+RefNo3_10R_34470A_60h_8.csv",
          'show': True,
          'parser': '34470A',
          'options': {
            'scaling': {
                "date": lambda data: (data["date"] - pd.Timestamp("2019-11-23 12:00:00", tz="UTC")).dt.total_seconds() / 3600,
                "value": lambda data: (data["value"] - data["value"][(data["date"]>=0) & (data["date"]<=24)].mean())/10,
            },
          },
        },
      ],
    },
    {
      'title': r' Stable Laser Systems VH 6020-4 (\qty{840}{\nm}) temperature over \qty{24}{\h}',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/stability_cavity.pgf"
      },
      #'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
      #"legend_position": "lower center",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "time",
          "y_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "temperature": {
                "label": r"Temperature",
                "color": colors[0],
            },
        },
      },
      'files': [
        {
          'filename': "Temperature Cavity-data-2023-05-04.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "temperature",
            },
            'scaling': {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
                "temperature": lambda data: data["temperature"] - 273.15
            },
          },
        },
      ],
    },
    {
      'title': r'Labkraken',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/kraken_inserts.pgf"
      },
      #'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
      #"legend_position": "lower center",
      'crop_secondary_to_primary': True,
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Number of daily database inserts",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "time",
          "y_scale": "log",
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "counts": {
                "label": r"\unit{inserts \per \day}",
                "color": colors[0],
            },
        },
      },
      'files': [
        {
          'filename': "kraken_database_counts.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "counts",
            },
            'scaling': {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
      ],
    },
    {
      'title': r'LabNode controller',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/labnode_performance.pgf"
      },
      #'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
      #"legend_position": "lower center",
      'crop_secondary_to_primary': True,
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": None,
          "x_scale": "time",
          "y_scale": "lin",
          "limits_y": [22.75, 23.25],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "temperature_in_loop": {
                "label": r"Temperature (in loop)",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        'show': True,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
          #"limits_y": [30.08,30.15],
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature_out_of_loop": {
                "label": r"Temperature (rack)",
                "color": colors[1],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "Neon Lab - 011-data-as-joinbyfield-2023-05-06 20_14_00.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "temperature_in_loop",
                7: "temperature_out_of_loop",
            },
            'scaling': {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
                "temperature_in_loop": lambda data: data["temperature_in_loop"].str.removesuffix(" °C").astype(float),
                "temperature_out_of_loop": lambda data: data["temperature_out_of_loop"].str.removesuffix(" °C").astype(float),
            },
          },
        },
      ],
    },
    {
      'title': "DgTemp Longterm",
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/dgTemp_longterm.pgf"
      },
      'crop': ['2019-07-06 00:00:00', '2019-07-12 16:00:00'], # Drift, Humidity? -> Yes
      "legend_position": "lower left",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Resistance deviation in \unit{\ohm}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -3,
          "x_scale": "time",
          "y_scale": "lin",
          #"limits_y": [22.75, 23.25],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "value_ext": {
                "label": r"CH1 (ab-precision RS2-10k)",
                "color": colors[5],
            },
            "value_int": {
                "label": r"CH2 (Internal reference)",
                "color": colors[3],
            },
        },
      },
      'secondary_axis': {
        'show': True,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Humidity in \unit{\percent RH}",
          "invert_x": False,
          "invert_y": True,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
          #"limits_y": [30.08,30.15],
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "humidity": {
                "label": r"Humdity",
                "color": colors[9],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "Rev2_INL_2019-07-02_08:08:20+00:00.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                1: "date",
                2: "value_ext",
                4: "value_int",
            },
            "scaling": {
                "value_ext": lambda x : (x["value_ext"] - x["value_ext"].mean()) / (2**31-1) * 4.096 / (50*10**-6) -14e-3,
                "value_int": lambda x : (x["value_int"] - x["value_int"].mean()) / (2**31-1) * 4.096 / (50*10**-6) +2e-3,
                #"value": lambda x : x["value"] / (2**31-1) * 4.096,# / (50*10**-6) - 25e-3,
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
        {
          'filename': "sensorData_2019-06-30 22_00_00_2019-07-18 06_34_00.csv",
          'show': True,
          'parser': 'smi',
          'options': {
            "sensor_id": 9,
            "label": "humidity",
            "scaling": {
            },
          },
        },
      ],
    },
    {
      'title': "DgTemp Performance",
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/dgTemp_laser_resonator.pgf"
      },
      #'crop': ['2018-10-17 10:00:00', '2020-10-16 06:00:00'],   # Air drafts (outer silicone)
      'crop': ['2018-10-15 12:00:00', '2020-10-16 06:00:00'],   # Air drafts (outer silicone)
      "legend_position": "upper right",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature deviation in \unit{\K}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -3,
          "x_scale": "time",
          "y_scale": "lin",
          #"limits_y": [22.75, 23.25],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "temperature": {
                "label": r"Laser resonator temperature",
                "color": colors[0],
            },
            "setpoint": {
                "label": r"Setpoint \qty{21.6893}{\celsius}",
                "color": colors[1],
                "linestyle": "dashed",
            },
        },
      },
      'files': [
        {
          'filename': "ADC_Serial_Read_2018-10-17_07:54:38+00:00.csv",
          'filename': "ADC_Serial_Read_2018-10-15_10:25:38+00:00.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "skiprows": 7,
            "columns": {
                1: "date",
                2: "adc_code",
            },
            "scaling": {
                "resistance": lambda x : x["adc_code"] / (2**31-1) * 4.096 / (50*10**-6),
                "temperature": lambda x : convertResistanceToTemperature(x["resistance"]) - convertResistanceToTemperature(300000000 * 4.096 / (2**31 - 1) / (50 * 10**-6)),
                "setpoint": lambda x : np.zeros(len(x)),
                #"value": lambda x : x["value"] / (2**31-1) * 4.096,# / (50*10**-6) - 25e-3,
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
        {
          'filename': "sensorData_2019-06-30 22_00_00_2019-07-18 06_34_00.csv",
          'show': False,
          'parser': 'smi',
          'options': {
            "sensor_id": 9,
            "label": "humidity",
            "scaling": {
            },
          },
        },
      ],
    },
    {
      'title': "DgTemp Testmass",
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/dgTemp_testmass.pgf"
      },
      'crop': ['2018-10-24 16:00:00', '2018-10-25 04:00:00'],
      "legend_position": "upper right",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature deviation in \unit{\K}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -6,
          "x_scale": "time",
          "y_scale": "lin",
          #"limits_y": [22.75, 23.25],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "temperature_mean": {
                "label": r"Temperature (Fluke 5611T-P)",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        'show': True,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Ambient temperature in  \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
          #"limits_y": [30.08,30.15],
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": r"Ambient temperature",
                "color": colors[1],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "HP3458A_GPIB_Read_2018-10-24_14:16:24+00:00.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "voltage",
            },
            "scaling": {
                "temperature_mass": lambda x : convertResistanceToTemperature(x["voltage"] / (50*10**-6)),
                "temperature_mean": lambda x : x["temperature_mass"] - x["temperature_mass"].mean() - 40e-6,
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
        {
          'filename': "fluke1524_2018-10-23_16:38:38+00:00.csv",   # Fixed direction. Going up now
          'show': True,
          'parser': 'fluke1524',
          'options': {
            "sensor_id": 2,  # Fluke Sensor 1 = Board Temp, Sensor 2 = Ambient
            "scaling": {
                "temperature": lambda x : x["temperature"] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "Vescent SliceQTC",
      'title': None,
      'show': False,
      #"output_file": {
      #  "fname": "../images/vescent_sliceqt.pgf"
      #},
      #'crop': ['2018-10-24 16:00:00', '2018-10-25 04:00:00'],
      "legend_position": "upper right",
      'crop_secondary_to_primary': True,
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature deviation in \unit{\K}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -3,
          "x_scale": "time",
          "y_scale": "lin",
          #"limits_y": [22.75, 23.25],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "reference_temperature": {
                "label": r"Temperature (SliceQTC)",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        'show': False,
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Ambient temperature in  \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "lin",
          "y_scale": "lin",
          "show_grid": False,
          #"limits_y": [30.08,30.15],
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": r"Ambient temperature",
                "color": colors[1],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': "SliceQTC_stability_2021-03-25_14:14:07+00:00.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "reference_temperature",
            },
            "scaling": {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
        {
          'filename': "fluke1524_2018-10-23_16:38:38+00:00.csv",   # Fixed direction. Going up now
          'show': True,
          'parser': 'fluke1524',
          'options': {
            "sensor_id": 2,  # Fluke Sensor 1 = Board Temp, Sensor 2 = Ambient
            "scaling": {
                "temperature": lambda x : x["temperature"] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "Room temperature Neon",
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/temperature_011_2016.pgf"
      },
      #'crop': ['2018-10-24 16:00:00', '2018-10-25 04:00:00'],
      "legend_position": "upper right",
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": 0,
          "x_scale": "time",
          "y_scale": "lin",
          #"limits_y": [22.75, 23.25],
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "temperature": {
                "label": r"Room temperature",
                "color": colors[0],
            },
        },
      },
      'files': [
        {
          'filename': "011_neon_temperature_2016-11 00-00-00-26_2016-11-27 00-00-00.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "temperature",
            },
            "scaling": {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
                "temperature": lambda data: data["temperature"] - 273.15,
            },
          },
        },
      ],
    },
    {
      'title': 'IRF9610 MOSFET simulation',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/mosfet_current_gate_bias.pgf"
      },
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Drain-Source Voltage $V_{DS}$ in \unit{\V}",
          'y_label': r"Drain Current $I_D$ in \unit{\A}",
          "invert_x": True,
          "invert_y": True,
          "fixed_order": -3,
          "y_scale": "linear",
        },
        'x-axis': "vds",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "Vgs3.5": {
                "label": "$V_{GS} = \qty{-3.5}{\V}$",
                "color": colors[0],
            },
            "Vgs4.0": {
                "label": "$V_{GS} = \qty{-4}{\V}$",
                "color": colors[1],
            },
            "Vgs4.5": {
                "label": "$V_{GS} = \qty{-4.5}{\V}$",
                "color": colors[2],
            },
            "Vgs5": {
                "label": "$V_{GS} = \qty{-5}{\V}$",
                "color": colors[3],
            },
            "isat": {
                "label": "$I_{sat}$",
                "color": colors[4],
                "linestyle": "--",
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'mosfet_current_source.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "vsd",
              1: "Vgs3.5",
              2: "Vgs4.0",
              3: "Vgs4.5",
              4: "Vgs5"
            },
            "scaling": {
              'vds': lambda x : -x["vsd"],
              #'isat': lambda x : calculate_saturation_current(x["vds"][x["vds"]>=-0.81], -0.813, -4/1000),
            },
          },
        },
      ],
    },
    {
      'title': 'Output impedance simulation',
      'title': None,
      'show': False,
      "output_file": {
          "fname": "../images/ltspice_output_impedance_simulation.pgf"
      },
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Drain-source voltage $V_{DS}$ in \unit{\V}",
          'y_label': r"Ouput Impedance $R_{out}$ in \unit{\ohm}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": 9,
          "y_scale": "log",
          #"x_scale": "log",  # Turn this on to show, that R_out is a power law
        },
        'x-axis': "vds",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "rout": {
                "label": r"DC",
                "color": colors[0],
            },
            "rout10MegHz": {
                "label": r"\qty{1}{\MHz}",
                "color": colors[1],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'mosfet_current_source_output_impedance.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "vload",
              1: "rout",
              2: "rout10MegHz"
            },
            "scaling": {
              "vds": lambda x : 3.5-x["vload"],
              "rout": lambda x : 10**(x["rout"]/20),
              "rout10MegHz": lambda x : 10**(x["rout10MegHz"]/20),
            },
          },
        },
      ],
    },
    {
      'title': r'DgDrive input filter simulation',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/input_filter_dgdrive.pgf"
      },
      "legend_position": "upper right",
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Magnitude in \unit{\V \per \V}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -3,
          "x_scale": "log",
          "y_scale": "log",
        },
        'x-axis': "freq",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "lc_filter": {
                "label": "Mag. LC only",
                "color": colors[0],
            },
            "cap_mult": {
                "label": "Mag. LC + C Mult.",
                "color": colors[1],
            },
        },
      },
      'secondary_axis': {
        'show': True,
        "axis_settings": {
          #'x_label': r"Time (UTC)",
          'y_label':  r"Impedance in \unit{\ohm}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -6,
          "x_scale": "log",
          "y_scale": "lin",
          "show_grid": False,
          #"limits_y": [22.4,23.4],
        },
        'x-axis': "freq",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "z_lc": {
                "label": r"$Z_{out}$ LC filter",
                "color": colors[2],
                "linestyle": "--",
            },
            "z_cap_mult": {
                "label": r"$Z_{out}$ C Mult.",
                "color": colors[4],
                "linestyle": "--",
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'input_filter_dgdrive.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "lc_filter",
              2: "cap_mult",
              3: "z_lc",
              4: "z_cap_mult"
            },
            "scaling": {
              'lc_filter': lambda x : 10**(x["lc_filter"]/20),
              'cap_mult': lambda x : 10**(x["cap_mult"]/20),
              'z_lc': lambda x : 10**(x["z_lc"]/20),
              'z_cap_mult': lambda x : 10**(x["z_cap_mult"]/20),
            },
          },
        },
      ],
    },
    {
      'title': 'Supply Filter Transfer function',
      'title': None,
      'show': False,
      "output_file": {
          "fname": "../images/dgDrive_supply_filter_bode.pgf"
      },
      "legend_position": "lower left",
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Magnitude in \unit{\dB}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": 9,
          #"y_scale": "lin",
          "x_scale": "log",
        },
        'x-axis': "frequency",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "magnitude": {
                "label": "LC Filter",
                "color": colors[1],
            },
            "lc_filter": {
                "label": "Simulation",
                "color": colors[0],
            },
        },
      },
      'files': [
        {
          'filename': 'DgDrive PSRR_take2_2023-02-18T01_12_08.csv',
          'show': True,
          'parser': 'bode100',
          'options': {
            "trace": 1,
            "columns": {
                0: "frequency",
                1: "magnitude",
            },
            "scaling": {
              "magnitude": lambda x : x["magnitude"]-40,
            },
          },
        },
        {
          'filename': 'input_filter_dgdrive.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "frequency",
                1: "lc_filter"
            },
            "scaling": {
              'lc_filter': lambda x : x["lc_filter"][(x["frequency"] >= 100) & (x["frequency"] <= 1e6)],
            },
          },
        },
      ],
    },
    {
      'title': 'Current Source noise (different R_filt)',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/current_source_noise_filter_resistors.pgf"
      },
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'crop_secondary_to_primary': True,
      "legend_position": "best",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Noise density in \unit[power-half-as-sqrt,per-mode=symbol]{\A \Hz\tothe{-0.5}}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          "x_scale": "log",
          "y_scale": "lin",
        },
        'x-axis': "frequency",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "249ohm": {
                "label": r"$R_{filt} = \qty{249}{\ohm}$",
                "color": colors[0],
            },
            "510ohm": {
                "label": r"$R_{filt} = \qty{510}{\ohm}$",
                "color": colors[1],
            },
            "1000ohm": {
                "label": r"$R_{filt} = \qty{1}{\kilo\ohm}$",
                "color": colors[2],
            },
            "1500ohm": {
                "label": r"$R_{filt} = \qty{1.5}{\kilo\ohm}$",
                "color": colors[3],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'current_regulator_v3_AD797_noise.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "frequency",
              1: "249ohm",
              2: "510ohm",
              3: "1000ohm",
              4: "1500ohm",
            },
            "scaling": {
            },
          },
        },
      ],
    },
    {
      'title': 'Output impedance Libbrecht & Hall current source',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/output_impedance_libbrecht_hall.pgf"
      },
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'crop_secondary_to_primary': True,
      "legend_position": "best",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Output impedance in \unit{\ohm}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": 6,
          "x_scale": "log",
          "y_scale": "log",
        },
        'x-axis': "frequency",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "noC": {
                "label": r"no $C_1$",
                "color": colors[0],
            },
            "1u": {
                "label": r"$C_1 = \qty{1}{\uF}$",
                "color": colors[1],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'modulation_input_LibrechtHall.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "frequency",
              1: "noC"
            },
            "scaling": {
                'noC': lambda x : 10**(x["noC"]/20),
            },
          },
        },
        {
          'filename': 'modulation_input_LibrechtHall_1u.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "frequency",
              1: "1u"
            },
            "scaling": {
                '1u': lambda x : 10**(x["1u"]/20),
            },
          },
        },
      ],
    },
    {
      'title': 'DgDrive Modulation Input',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/dgDrive_modulation_input.pgf"
      },
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'crop': {
          "crop_index": "frequency",
          "crop": [1e2, 5e6],
      },
      "legend_position": "best",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Amplitude in \unit{\decibel}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": None,
          "x_scale": "log",
          "y_scale": "lin",
        },
        'x-axis': "frequency",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "amplitude_normalized": {
                "label": r"Normalized amplitude",
                "color": colors[0],
            },
            "3dB": {
                "label": r"\qty{\pm 3}{\decibel}",
                "color": colors[1],
                "linestyle": (0, (5, 10)),  # loosely dashed
            },
            "-3dB": {
                "label": None,
                "color": colors[1],
                "linestyle": (0, (5, 10)),  # loosely dashed
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'secondary_axis': {
        "show": True,
        "axis_settings": {
          'y_label': r"Phase in \unit{\degree}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": None,
          "x_scale": "log",
          "y_scale": "lin",
          "show_grid": False,
        },
        'x-axis': "frequency",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "phase_corrected": {
                "label": r"Phase",
                "color": colors[2],
            },
        },
      },
      'files': [
        {
          'filename': 'DgDrive_modulation_input.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              1: "frequency",
              3: "amplitude",
              4: "phase",
              5: "reference",
              6: "reference_phase",
            },
            "scaling": {
                "amplitude_corrected": lambda x : x["amplitude"] - x["reference"],
                "amplitude_normalized": lambda x : x["amplitude_corrected"] - np.mean(x["amplitude_corrected"][(x["frequency"] >= 1e3) & (x["frequency"] <= 1e5)]),
                "3dB": lambda x : np.repeat(np.mean(x["amplitude_normalized"][(x["frequency"] >= 1e3) & (x["frequency"] <= 1e5)]+3), len(x)),
                "-3dB": lambda x : np.repeat(np.mean(x["amplitude_normalized"][(x["frequency"] >= 1e3) & (x["frequency"] <= 1e5)]-3), len(x)),
                "phase_corrected": lambda x : x["phase"] - x["reference_phase"],
            },
          },
        },
      ],
    },
    {
      'title': 'DgDrive Output Impedance',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/dgDrive_output_impedance_dc.pgf"
      },
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      #'crop': {
      #    "crop_index": "frequency",
      #    "crop": [1e2, 5e6],
      #},
      "legend_position": "best",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time in \unit{\s}",
          'y_label': r"Output current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          "x_scale": "lin",
          "y_scale": "lin",
        },
        'x-axis': "time",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "value": {
                "label": None,
                "color": colors[0],
            },
            "lower": {
                "label": None,
                "color": colors[1],
            },
            "upper": {
                "label": None,
                "color": colors[1],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'DgDrive_output_impedance_3M3_10PLC_AZ.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "time",
              1: "value",
            },
            "skiprows": 2,
            "scaling": {
                "time": lambda x : x["time"] / 2.5,#  NPLC=10 + autozero
                "lower": lambda x :np.where(x["time"] <= 30.5, np.mean(x["value"][x["time"] <= 30.5]), np.nan),
                "upper": lambda x : np.where(x["time"] > 30.5, np.mean(x["value"][x["time"] > 30.5]), np.nan),
            },
          },
        },
      ],
    },
  ]

  plots = (plot for plot in plots if plot.get('show', True))
  for plot in plots:
    print("Ploting {plot!s}".format(plot=plot['title']))
    plot_series(plot=plot)
