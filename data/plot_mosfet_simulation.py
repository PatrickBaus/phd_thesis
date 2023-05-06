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

def prepare_axis(ax, label, invert_yaxis, color_map=None, fixed_order=None):
  if fixed_order is not None:
    ax.yaxis.set_major_formatter(FixedOrderFormatter(fixed_order, useOffset=True))
  else:
    ax.yaxis.get_major_formatter().set_useOffset(False)

  #ax.set_yscale('log')
  if invert_yaxis:
    ax.invert_yaxis()
  ax.invert_xaxis()

  if color_map is not None:
    ax.set_prop_cycle('color', color_map)

def plot_data(ax, data, x_axis, column_settings):
  for column, settings in column_settings.items():
      if column in data:
          ax.plot(data[x_axis], data[column], color=settings["color"], marker="", label=settings["label"], alpha=0.7, linewidth=settings.get("linewidth", 1))

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
        fixed_order=plot_settings['axis_fixed_order'],
        label=plot_settings['label'],
        invert_yaxis=plot_settings.get("invert_yaxis", False),
        color_map=plt.cm.tab10.colors
      )

    plot_data(ax1, data, x_axis=plot_settings["x-axis"], column_settings=plot_settings['columns_to_plot'])

    ax1.set_ylabel(plot_settings["label"])
    ax1.set_xlabel(r"Drain Current $I_{D}$ in \unit{\A}")
    lines, labels = ax1.get_legend_handles_labels()

    plot_settings = plot.get('secondary_axis', {})
    if plot_settings.get("show", False):
      ax2 = ax1.twinx()
      plot_data(ax2, data, plot_settings['columns_to_plot'])

      ax2.set_ylabel(plot_settings["label"])

      lines2, labels2 = ax2.get_legend_handles_labels()
      lines += lines2
      labels += labels2
    plt.legend(lines, labels, loc="upper left")

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
      'title': '2N5460 JFET simulation',
      'title': None,
      'show': False,
      'crop_secondary_to_primary': True,
      'primary_axis': {
        'label': r"Drain Current $I_D$ in \unit{\A}",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'invert_yaxis': True,
        'axis_fixed_order': -3,
        'columns_to_plot': {
            "Vgs0.0": {
                "label": "$V_{GS} = \qty{0}{\V}$",
                "color": colors[0],
            },
            "Vgs0.5": {
                "label": "$V_{GS} = \qty{0.5}{\V}$",
                "color": colors[1],
            },
            "Vgs1.0": {
                "label": "$V_{GS} = \qty{1}{\V}$",
                "color": colors[2],
            },
            "Vgs1.5": {
                "label": "$V_{GS} = \qty{1.5}{\V}$",
                "color": colors[3],
            },
            "Vgs2.0": {
                "label": "$V_{GS} = \qty{2}{\V}$",
                "color": colors[4],
            },
            "Vgs2.5": {
                "label": "$V_{GS} = \qty{2.5}{\V}$",
                "color": colors[5],
            },

        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'jfet_current_source.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": ["Vgs0.0", "Vgs0.5", "Vgs1.0", "Vgs1.5", "Vgs2.0", "Vgs2.5"],
            "scaling": {
              'Vds': lambda x : -x["Vsd"],
            },
          },
        },
      ],
    },
    {
      'title': 'IRF9610 MOSFET simulation',
      'title': None,
      'show': True,
      #"output_file": {
      #  "fname": "../images/temperature_011_2016.pgf"
      #},
      'crop_secondary_to_primary': True,
      'primary_axis': {
        'label': r"Drain Current $I_D$ in \unit{\A}",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'invert_yaxis': True,
        'axis_fixed_order': -3,
        'x-axis': "Vds",
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

        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'mosfet_current_source.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": ["Vsd", "Vgs3.5", "Vgs4.0", "Vgs4.5", "Vgs5"],
            "scaling": {
              'Vds': lambda x : -x["Vsd"],
            },
          },
        },
      ],
    },
    {
      'title': 'MOSFET Id LTSpice example',
      'title': None,
      'show': False,
      'crop_secondary_to_primary': True,
      'primary_axis': {
        'label': r"Drain Current $I_D$ in \unit{\A}",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'invert_yaxis': True,
        'axis_fixed_order': -3,
        'columns_to_plot': {
            "Vgs0.2": {
                "label": "$V_{GS} + V_{th} = \qty{-0.2}{\V}$",
                "color": colors[0],
            },
            "Vgs0.4": {
                "label": "$V_{GS} + V_{th} = \qty{-0.4}{\V}$",
                "color": colors[1],
            },
            "Vgs0.6": {
                "label": "$V_{GS} + V_{th} = \qty{-0.6}{\V}$",
                "color": colors[2],
            },
            "Vgs0.8": {
                "label": "$V_{GS} + V_{th} = \qty{-0.8}{\V}$",
                "color": colors[3],
            },

        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'mosfet_id.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": [f"Vgs{val:.1f}" for val in np.arange(0.2, 0.9, 0.2)],
            "scaling": {
              'Vds': lambda x : -x["Vsd"],
            },
          },
        },
      ],
    },
    {
      'title': 'MOSFET gm LTSpice example',
      'title': None,
      'show': True,
      'crop_secondary_to_primary': True,
      'primary_axis': {
        'label': r"Transconductance $g_m$ in \unit{\siemens}",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'invert_yaxis': False,
        'axis_fixed_order': -3,
        'x-axis': "Id",
        'columns_to_plot': {
            "gm": {
                "label": r"$g_m$",
                "color": colors[0],
            },

        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': 'mosfet_gm.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": ["Vsg", "Id", "gm"],
            "scaling": {
              "gm": lambda x : -x["gm"],
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
