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
    "savefig.directory": os.chdir(os.path.dirname(__file__)),
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
        x_data = pd.to_datetime(x_data).astype(dtype)

    return x_data, y_data

def process_data(data, columns, plot_type):
    if plot_type=='relative':
        data[columns] = data[columns] - data[columns].mean().tolist()
    elif plot_type=='proportional':
        data[columns] = data[columns] / data[columns].iloc[:30].mean().tolist() - 1

def prepare_axis(ax, axis_settings, color_map=None):
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

  if color_map is not None:
    ax.set_prop_cycle('color', color_map)

def plot_data(ax, data, x_axis, column_settings):
  for column, settings in column_settings.items():
      if column in data:
          #x_data, y_data = downsample_data(data[x_axis], data[column])
          x_data, y_data = data[x_axis], data[column]
          ax.plot(
              x_data,
              y_data,
              marker="",
              alpha=0.7,
              **settings
          )
          #ax.plot(data[x_axis], data[column], color=settings["color"], marker="", label=settings["label"], alpha=0.7, linewidth=settings.get("linewidth", 1))

def plot_series(plot):
  # Load the data to be plotted
  plot_files = (plot_file for plot_file in plot['files'] if plot_file.get('show', True))
  data = pd.concat(
    (load_data(plot_file)[0] for plot_file in plot_files),
    sort=True
  )

  # If we have something to plot, proceed
  if not data.empty:
    crop_data(data, crop_index="date", crop=plot.get('crop'))

    plot_settings = plot['primary_axis']
    process_data(data=data, columns=plot_settings['columns_to_plot'], plot_type=plot_settings.get('plot_type','absolute'))

    ax1 = plt.subplot(111)
    #plt.tick_params('x', labelbottom=False)
    prepare_axis(
        ax=ax1,
        axis_settings=plot_settings["axis_settings"],
        color_map=plt.cm.tab10.colors
      )

    plot_data(ax1, data,  plot_settings["x-axis"], plot_settings['columns_to_plot'])

    lines, labels = ax1.get_legend_handles_labels()

    plot_settings = plot.get('secondary_axis', {})
    if plot_settings.get("show", False):
      ax2 = ax1.twinx()
      prepare_axis(
        ax=ax2,
        axis_settings=plot_settings["axis_settings"],
        color_map=plt.cm.tab10.colors
      )
      ax2.format_coord = make_format(ax2, ax1)
      plot_data(ax2, data, plot_settings["x-axis"], plot_settings['columns_to_plot'])

      lines2, labels2 = ax2.get_legend_handles_labels()
      lines += lines2
      labels += labels2
    plt.legend(lines, labels, loc=plot.get("legend_position", "upper left"))

  fig = plt.gcf()
#  fig.set_size_inches(11.69,8.27)   # A4 in inch
#  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
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
      'title': r'SMC11 stability over \qty{60}{\hour}',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/stability_smc11.pgf"
      },
      'crop': ['2019-12-01 00:00:00', '2019-12-02 00:00:00'],
      #"legend_position": "lower center",
      'crop_secondary_to_primary': True,
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -6,
          "x_scale": "time",
          "y_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "value": {
                "label": r"SMC11",
                "color": colors[0],
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
          'filename': "SMC11_50mA+10R_60h_2.csv",
          'show': True,
          'parser': '34470A',
          'options': {
            'scaling': {
                "value": lambda data: (data["value"] - data["value"].mean())/10,
            },
          },
        },
      ],
    },
    {
      'title': r'DgDrive stability over \qty{60}{\hour}',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/stability_dgDrive.pgf"
      },
      'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
      #"legend_position": "lower center",
      'crop_secondary_to_primary': True,
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -6,
          "x_scale": "time",
          "y_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',
        'columns_to_plot': {
            "value": {
                "label": r"DgDrive-500-LN",
                "color": colors[0],
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
          'filename': "DgDrive_390uLP_caseGND_50mA_100h.csv",
          'show': True,
          'parser': '34470A',
          'options': {
            'scaling': {
                "value": lambda data: (data["value"] - data["value"].mean())/10 +0.02*10**-6,
            },
          },
        },
      ],
    },
    {
      'title': r'DgDrive stability over \qty{60}{\hour}',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/stability_cavity.pgf"
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
          'filename': "Temperature Cavity-data-2023-05-05 20_15_08.csv",
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "temperature",
            },
            'scaling': {
                "date": lambda data: pd.to_datetime(data.date, utc=True),
                "temperature": lambda data: data["temperature"].str.removesuffix(" °C").astype(float)
            },
          },
        },
      ],
    },
    {
      'title': r'Labkraken',
      'title': None,
      'show': True,
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
  ]

  plots = (plot for plot in plots if plot.get('show', True))
  for plot in plots:
    print("Ploting {plot!s}".format(plot=plot['title']))
    plot_series(plot=plot)
