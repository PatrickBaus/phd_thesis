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
from scipy.constants import elementary_charge, k as kB
from scipy import integrate
import seaborn as sns

pd.plotting.register_matplotlib_converters()

from file_parser import parse_file

colors = sns.color_palette("colorblind", 11)

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

def crop_data(data, crop_index=None, crop=None):
    if crop_index is not None:
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

def plotShotNoise(ax, current):
  x = np.sqrt(2 * elementary_charge * current)

  ax.axhline(x, color='r', linestyle='--', label="Shot noise, {current} A".format(current=current))

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
  if axis_settings.get("invert_y"):
    ax.invert_yaxis()
  if axis_settings.get("invert_x"):
    ax.invert_xaxis()

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
          ax.plot(
            data[x_axis][~np.isnan(data[column])],
            data[column][~np.isnan(data[column])],
            color=settings["color"],
            marker="",
            label=settings["label"],
            alpha=0.7,
            linewidth=settings.get("linewidth", 1),
            linestyle=settings.get("linestyle", "-"),
            zorder=settings.get("zorder", None),
          )

def integrate_data(data, x_axis, column_settings):
    print("  Integrated current noise:")
    for column, settings in column_settings.items():
        if column in data:
            current_data = data.dropna(subset=column)
            current_data_100khz = data.dropna(subset=column)[(current_data[x_axis] >= 10**1) & (current_data[x_axis] <= 10**5)]
            current_data = current_data[(current_data[x_axis] >= 10**1) & (current_data[x_axis] <= 10**6)]
            rms_100khz = integrate.trapezoid(current_data_100khz[column]**2, current_data_100khz[x_axis])
            rms = integrate.trapezoid(current_data[column]**2, current_data[x_axis])
            print(f"    {column}: {np.min(current_data_100khz[x_axis])} Hz - {np.max(current_data_100khz[x_axis])} kHz, {np.sqrt(rms_100khz):.2e} A_rms; {np.min(current_data[x_axis])} Hz - {np.max(current_data[x_axis])} kHz, {np.sqrt(rms):.2e} A_rms")

def plot_series(plot):
  # Load the data to be plotted
  plot_files = (plot_file for plot_file in plot['files'] if plot_file.get('show', True))
  data = pd.concat(
    (load_data(plot_file)[0] for plot_file in plot_files),
    sort=True
  )

  # If we have something to plot, proceed
  if not data.empty:
    crop_data(data, **plot.get('crop', {}))

    plot_settings = plot['primary_axis']
    process_data(data=data, columns=plot_settings['columns_to_plot'], plot_type=plot_settings.get('plot_type','absolute'))

    ax1 = plt.subplot(111)
    #plt.tick_params('x', labelbottom=False)
    prepare_axis(
        ax=ax1,
        axis_settings=plot_settings["axis_settings"],
        color_map=plt.cm.tab10.colors
      )

    x_axis = plot_settings["x-axis"]
    integrate_data(data, x_axis=x_axis, column_settings=plot_settings['columns_to_plot'])
    plot_data(ax1, data, x_axis=x_axis, column_settings=plot_settings['columns_to_plot'])

    #plotShotNoise(ax1, 0.5)
    #plotShotNoise(ax1, 0.1)
    #plotShotNoise(ax1, 0.02)

    lines, labels = ax1.get_legend_handles_labels()

    plot_settings = plot.get('secondary_axis', {})
    if plot_settings.get("show", False):
      ax2 = ax1.twinx()
      prepare_axis(
          ax=ax2,
          axis_settings=plot_settings["axis_settings"],
          color_map=plt.cm.tab10.colors
      )

      plot_data(ax2, data, x_axis=x_axis, column_settings=plot_settings['columns_to_plot'])

      #ax2.set_ylabel(plot_settings["label"])

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
    #fig.set_size_inches(441.01773 / 72.27 * 0.8 / phi, 441.01773 / 72.27 * 0.8)  # landscape
    phi = (5**.5-1) / 2  # golden ratio
    fig.set_size_inches(441.01773 / 72.27 * 0.9, 441.01773 / 72.27 * 0.9 * phi)  # thesis
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
      'title': 'DgDrive Noise comparison',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/laser_driver_noise_measurement.pgf"
      },
      #'crop': {
      #    "crop_index": "frequency",
      #    "crop": [1e2, 5e6],
      #},
      "legend_position": "upper right",
      "plot_size": (441.01773 / 72.27 * 0.8 / phi, 441.01773 / 72.27 * 0.8),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Noise density in \unit[power-half-as-sqrt,per-mode=symbol]{\A \Hz\tothe{-0.5}}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -9,
          "x_scale": "log",
          "y_scale": "log",
          #"y_scale": "lin",
        },
        'x-axis': "freq",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "toptica_dcc": {
                "label": "Toptica DCC 110",
                "color": colors[9],
            },
            "lqo": {
                "label": "LQO LQprO-140",
                "color": colors[1],
            },
            "moglabs": {
                "label": "Moglabs DLC-202",
                "color": colors[2],
            },
            "vescent": {
                "label": "Vescent D2-105-500 (no display)",
                "color": colors[5],
            },
            "dgDrive": {
                "label": "DgDrive-500-LN v2.3.0",
                "color": colors[3],
                "zorder": 2.02,
            },
            "dgDrive_simulation": {
                "label": "LTSpice simulation (DgDrive)",
                "color": "black",
                "zorder": 2.03,
            },
            "lna_background": {
                "label": r"LNA background (\qty{10}{\ohm})",
                "color": colors[0],
            },
            "smc11": {
                "label": "Sisyph SMC11 (\qty{470}{\mA})",
                "color": colors[4],
                "zorder": 2.01,
            },
            "tia_background": {
                "label": r"SR560 background (\qty{1}{\kilo\ohm})",
                "color": colors[7],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': './current_source_noise/lna_background.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "lna_background",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/dgDrive-500_2-3-0.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "dgDrive",
            },
            "scaling": {
              #"dgDrive": lambda x: 20*np.log10(x["dgDrive"])
            },
          },
        },
        {
          'filename': './current_source_noise/tia_background_1k.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "tia_background",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/toptica_dcc_110.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "toptica_dcc",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/moglabs_dlc_202.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "moglabs",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/vescent_d2-105-500_no_display.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "vescent",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/smc11.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "smc11",
            },
            "scaling": {
                #"smc11": lambda x: 20*np.log10(x["smc11"])
            },
          },
        },
        {
          'filename': './current_source_noise/LQprO-140.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "lqo",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': 'current_regulator_v3_AD797+TIA_simple.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "delimiter": "\t",
            "columns": {
              0: "freq",
              1: "dgDrive_simulation"
            },
            "scaling": {
            },
          },
        },
      ],
    },
    {
      'title': 'DgDrive vs HMP4040',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/laser_driver_noise_hmp4040.pgf"
      },
      #'crop': {
      #    "crop_index": "frequency",
      #    "crop": [1e2, 5e6],
      #},
      "legend_position": "upper right",
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Noise density in \unit[power-half-as-sqrt,per-mode=symbol]{\A \Hz\tothe{-0.5}}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -9,
          "x_scale": "log",
          "y_scale": "log",
          #"y_scale": "lin",
        },
        'x-axis': "freq",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "dgDrive": {
                "label": "DgDrive-500-LN v2.3.0",
                "color": colors[3],
            },
            "dgDrive_hmp4040": {
                "label": "DgDrive-500-LN v2.1.0 (HMP4040)",
                "color": colors[10],
            },
            "dgDrive_simulation": {
                "label": "LTSpice simulation (DgDrive)",
                "color": "black",
            },
            "shot_noise_200mA": {
                "label": r"Shot noise, \qty{200}{\mA}",
                "color": "red",
                "linestyle": "dotted",
                "linewidth": 1.5,
            },
            "shot_noise_100mA": {
                "label": r"Shot noise, \qty{100}{\mA}",
                "color": "red",
                "linestyle": "dashed",
                "linewidth": 1.5,
            },
            "shot_noise_20mA": {
                "label": r"Shot noise, \qty{20}{\mA}",
                "color": "red",
                "linestyle": "dashdot",
                "linewidth": 1.5,
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': './current_source_noise/dgDrive-500_2-3-0.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "dgDrive",
            },
            "scaling": {
              #"dgDrive": lambda x: 20*np.log10(x["dgDrive"])
            },
          },
        },
        {
          'filename': './current_source_noise/dgDrive-500_2-1-0_hmp4040.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "dgDrive_hmp4040",
            },
            "scaling": {
              #"dgDrive": lambda x: 20*np.log10(x["dgDrive"])
            },
          },
        },
        {
          'filename': 'current_regulator_v3_AD797+TIA_simple.txt',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "delimiter": "\t",
            "columns": {
              0: "freq",
              1: "dgDrive_simulation"
            },
            "scaling": {
                "shot_noise_100mA": lambda data: np.ones_like(data["freq"]) * np.sqrt(2 * elementary_charge * 0.1),
                "shot_noise_200mA": lambda data: np.ones_like(data["freq"]) * np.sqrt(2 * elementary_charge * 0.18),
                "shot_noise_20mA": lambda data: np.ones_like(data["freq"]) * np.sqrt(2 * elementary_charge * 0.02),
            },
          },
        },
      ],
    },
    {
      'title': 'Vescent D2-105-500 gain peaking',
      'title': None,
      'show': False,
      "output_file": {
        "fname": "../images/vescent_gain_peaking.pgf"
      },
      #'crop': {
      #    "crop_index": "frequency",
      #    "crop": [1e2, 5e6],
      #},
      "legend_position": "lower left",
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Noise density in \unit[power-half-as-sqrt,per-mode=symbol]{\A \Hz\tothe{-0.5}}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -9,
          "x_scale": "log",
          "y_scale": "log",
          #"y_scale": "lin",
        },
        'x-axis': "freq",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "vescent": {
                "label": "Vescent D2-105-500 (\qty{50}{\mA}, $V_{DS} = \qty{9.8}{\V}$)",
                "color": colors[3],
            },
            "vescent_300mA": {
                "label": "Vescent D2-105-500 (\qty{300}{\mA}, $V_{DS} = \qty{4.0}{\V}$)",
                "color": colors[0],
            },
            "vescent_400mA": {
                "label": r"Vescent D2-105-500 (\qty{400}{\mA}, $V_{DS} = \qty{1.7}{\V}$)",
                "color": colors[4],
            },
            "vescent_450mA": {
                "label": "Vescent D2-105-500 (\qty{450}{\mA}, $V_{DS} = \qty{0.5}{\V}$)",
                "color": colors[2],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': './current_source_noise/vescent_d2-105-500_no_display_2kHz.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "vescent",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/vescent_d2-105-500_300mA.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "vescent_300mA",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/vescent_d2-105-500_400mA.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "vescent_400mA",
            },
            "scaling": {
            },
          },
        },
        {
          'filename': './current_source_noise/vescent_d2-105-500_450mA.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "vescent_450mA",
            },
            "scaling": {
            },
          },
        },
      ],
    },
{
      'title': 'DgTemp 1.0.0, noise floor',
      'title': None,
      'show': True,
      #"output_file": {
      #  "fname": "../images/dgTemp_244sps_shorted_input.pgf"
      #},
      #'crop': {
      #    "crop_index": "frequency",
      #    "crop": [1e2, 5e6],
      #},
      "legend_position": "upper right",
      "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Frequency in \unit{\Hz}",
          'y_label': r"Noise density in \unit[power-half-as-sqrt,per-mode=symbol]{\V \Hz\tothe{-0.5}}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": -9,
          "x_scale": "log",
          "y_scale": "log",
          #"y_scale": "lin",
        },
        'x-axis': "freq",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "dgTemp": {
                "label": "DgTemp v1.0.0, 244 Hz",
                "color": colors[0],
            },
        },
        'filter': None,#filter_savgol(window_length=101, polyorder=3),
      },
      'files': [
        {
          'filename': './dgTemp_noise/dgTemp_244sps_shorted_input.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
              0: "freq",
              1: "dgTemp",
            },
            "scaling": {
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
