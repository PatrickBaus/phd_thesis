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

def calculate_saturation_current(vds, kappa, ld):
    return 0.5 * kappa * vds**2 * (1 + ld * vds)

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
                .format(*['({}, {:.6E})'.format(matplotlib.dates.num2date(x).strftime("%H:%M:%S"), y) for x,y in coords]))
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

def process_data(data, columns, plot_type):
    if plot_type=='relative':
        data[columns] = data[columns] - data[columns].mean().tolist()
    elif plot_type=='proportional':
        data[columns] = data[columns] / data[columns].iloc[:30].mean().tolist() - 1

def prepare_axis(ax, axis_settings, color_map=None):
  if axis_settings.get("y_fixed_order") is not None:
      ax.yaxis.set_major_formatter(FixedOrderFormatter(axis_settings["y_fixed_order"], useOffset=True))
  else:
      ax.yaxis.get_major_formatter().set_useOffset(False)

  if axis_settings.get("x_fixed_order") is not None:
      ax.xaxis.set_major_formatter(FixedOrderFormatter(axis_settings["x_fixed_order"], useOffset=True))
  else:
      ax.xaxis.get_major_formatter().set_useOffset(False)

  if axis_settings.get("y_scale") == "log":
    ax.set_yscale('log')
  if axis_settings.get("x_scale") == "log":
    ax.set_xscale('log')
  if axis_settings.get("invert_y"):
    ax.invert_yaxis()
  if axis_settings.get("invert_x"):
    ax.invert_xaxis()

  ax.grid(True, which="minor", ls="-", color='0.85')
  ax.grid(True, which="major", ls="-", color='0.45')

  ax.set_ylabel(axis_settings["y_label"])
  ax.set_xlabel(axis_settings["x_label"])

  if color_map is not None:
    ax.set_prop_cycle('color', color_map)

def plot_data(ax, data, x_axis, column_settings):
  shared_bins = np.histogram_bin_edges(data[column_settings.keys()], bins="sturges")
  for column, settings in column_settings.items():
      if column in data:
          if "bins" in settings:
              n, bins, _ = ax.hist(
                  data[column],
                  alpha=0.7,
                  **settings
              )
          else:
              n, bins, _ = ax.hist(
                  data[column],
                  alpha=0.7,
                  bins=shared_bins,
                  **settings
              )
          # To calculate the probability, calculate the one we do want and subtract it from 1, because the very high
          # impedances are ignored (see range option) for readability
          print(f"Smallest bin with more than zero counts: {bins[:-1][n > 0][0]/1e6} MΩ")
          print(f"Probability of getting more than 7.5 MΩ: {1-(sum(n[bins[:-1]<7.5e6]) / len(data[column]))}")

def plot_series(plot):
  # Load the data to be plotted
  plot_files = (plot_file for plot_file in plot['files'] if plot_file.get('show', True))
  data = pd.concat(
    (load_data(plot_file)[0] for plot_file in plot_files),
    sort=True
  )
  # Removes NAs from each column by shifting the values up, then remove all rows, that have no data
  data = data.apply(lambda x: pd.Series(x.dropna().values)).dropna()

  # If we have something to plot, proceed
  if not data.empty:
    crop_data(data, crop_index="Rout", crop=plot.get('crop'))

    plot_settings = plot['primary_axis']
    process_data(data=data, columns=plot_settings['columns_to_plot'], plot_type=plot_settings.get('plot_type','absolute'))

    ax1 = plt.subplot(111)
    #plt.tick_params('x', labelbottom=False)
    prepare_axis(
        ax=ax1,
        axis_settings=plot_settings["axis_settings"],
        color_map=plt.cm.tab10.colors
      )

    plot_data(ax1, data, x_axis=plot_settings["x-axis"], column_settings=plot_settings['columns_to_plot'])

    lines, labels = ax1.get_legend_handles_labels()

    plot_settings = plot.get('secondary_axis', {})
    if plot_settings.get("show", False):
      ax2 = ax1.twinx()
      plot_data(ax2, data, plot_settings['columns_to_plot'])

      ax2.set_ylabel(plot_settings["label"])

      lines2, labels2 = ax2.get_legend_handles_labels()
      lines += lines2
      labels += labels2
    plt.legend(lines, labels, loc="best")

  fig = plt.gcf()
#  fig.set_size_inches(11.69,8.27)   # A4 in inch
#  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
  phi = (5**.5-1) / 2  # golden ratio
  if plot.get("plot_size"):
      fig.set_size_inches(*plot["plot_size"])
  else:
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
      'title': 'Parallel IRF9610 MOSFET Monto-Carlo Simulation',
      'title': None,
      'show': False,
      "output_file": {
          "fname": "../images/ltspice_mosfet_mc_output_impedance.pgf"
      },
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Output impedance in \unit{\ohm}",
          'y_label': r"Counts",
          "invert_x": False,
          "invert_y": False,
          "y_fixed_order": None,
          "x_scale": "linear",
        },
        'x-axis': "num",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "Rout": {
                "label": "Parallel MOSFETs",
                "color": colors[1],
                #"bins": 50,
            },
            "Rout_s": {
                "label": "Single MOSFET",
                "color": colors[2],
                #"bins": auto,
            },
            "Rout_p_sigma": {
                "label": r"Parallel MOSFET $V_{DS}+1\sigma$",
                "color": colors[0],
                #"bins": auto,
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'mosfet_current_source_parallel_mc.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {0: "num", 1: "Rout"},
            "scaling": {
              "Rout": lambda x : 10**(x["Rout"]/20),
            },
          },
        },
        {
          'filename': 'mosfet_current_source_single_mc.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {0: "num", 1: "Rout_s"},
            "scaling": {
              "Rout_s": lambda x : 10**(x["Rout_s"]/20),
            },
          },
        },
        {
          'filename': 'mosfet_current_source_parallel_mc-sigma.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {0: "num", 1: "Rout_p_sigma"},
            "scaling": {
              "Rout_p_sigma": lambda x : 10**(x["Rout_p_sigma"]/20),
            },
          },
        },
      ],
    },
    {
      'title': 'Howland Current Source Output Impedance',
      'title': None,
      'show': False,
      "output_file": {
          "fname": "../images/ltspice_howland_mc_output_impedance.pgf"
      },
      'crop_secondary_to_primary': True,
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Resistance in \unit{\ohm}",
          'y_label': r"Normalised resistance density in \unit{\per \ohm}",
          "invert_x": False,
          "invert_y": False,
          "x_fixed_order": 6,
          "x_scale": "linear",
        },
        'x-axis': "num",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "Rout005": {
                "label": r"HCS, \qty{0.05}{\percent} tolerance",
                "color": colors[0],
                "density": True,
                "bins": 100,
                "range": (-1e8, 1e8),
                "range": (0, 5e7),
            },
            "Rout001": {
                "label": r"HCS, \qty{0.01}{\percent} tolerance",
                "color": colors[1],
                "density": True,
                "bins": 100,
                "range": (0, 5e7),
            },
            "Routi005": {
                "label": r"Improved HCS, \qty{0.05}{\percent} tolerance",
                "color": colors[3],
                "density": True,
                "bins": 100,
                "range": (0, 5e7),
            },
            "Routi001": {
                "label": r"Improved HCS, \qty{0.01}{\percent} tolerance",
                "color": colors[2],
                "density": True,
                "bins": 100,
                "range": (0, 5e7),
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'howland_current_source_001.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "delimiter": "\t",
            "columns": {0: "num", 1: "Rout001"},
            "scaling": {
                "Rout001": lambda x: np.abs(x["Rout001"])
            },
          },
        },
        {
          'filename': 'howland_current_source_005.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "delimiter": "\t",
            "columns": {0: "num", 1: "Rout005"},
            "scaling": {
                "Rout005": lambda x: np.abs(x["Rout005"])
            },
          },
        },
        {
          'filename': 'improved_howland_current_source_001.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "delimiter": "\t",
            "columns": {0: "num", 1: "Routi001"},
            "scaling": {
                "Rout005": lambda x: np.abs(x["Routi001"])
            },
          },
        },
        {
          'filename': 'improved_howland_current_source_005.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "delimiter": "\t",
            "columns": {0: "num", 1: "Routi005"},
            "scaling": {
                "Rout005": lambda x: np.abs(x["Routi005"])
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
