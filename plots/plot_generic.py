#!/usr/bin/env python
import argparse
import ast
import glob
import importlib

import matplotlib
import matplotlib.pyplot as plt
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

__version__ = "0.9.0"

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
    if crop is not None:
        data.sort_values(by=crop_index, inplace=True)
        index_to_drop = data[(data[crop_index] < crop[0]) | (data[crop_index] > crop[1])].index if len(crop) > 1 else data[data[crop_index] < crop[0]].index
        data.drop(index_to_drop , inplace=True)

    #y = 1/(0.000858614 + 0.000259555 * np.log(y) + 1.35034*10**-7 * np.log(y)**3)
    #if "date" in data:
    #    print(f"    Begin date: {data.date.iloc[0].tz_convert('Europe/Berlin')}")
    #    print(f"    End date:   {data.date.iloc[-1].tz_convert('Europe/Berlin')} (+{(data.date.iloc[-1]-data.date.iloc[0]).total_seconds()/3600:.1f} h)")

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

def process_data(data, columns, plot_type):
    if plot_type=='relative':
        data[columns] = data[columns] - data[columns].mean().tolist()
    elif plot_type=='proportional':
        data[columns] = data[columns] / data[columns].iloc[:30].mean().tolist() - 1

def prepare_axis(ax, axis_settings):
  if axis_settings.get("y_scale") == "log":
    ax.set_yscale('log')
  if axis_settings.get("x_scale") == "log":
    ax.set_xscale('log')
  if axis_settings.get("x_scale") == "time":
      ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
      if axis_settings.get("date_format"):
          ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(axis_settings["date_format"]))
      else:
          ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
  if axis_settings.get("x_scale") == "timedelta":
      ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(0,48,3)))
      #ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H"))

  if axis_settings.get("fixed_order") is not None:
    ax.yaxis.set_major_formatter(FixedOrderFormatter(axis_settings["fixed_order"], useOffset=True))
  else:
    try:
        ax.yaxis.get_major_formatter().set_useOffset(False)
    except AttributeError:
        pass  # The log formatter does not have set_useOffset()

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
            print(f"  Integrated noise: {np.sqrt(np.mean(data[column]**2))}")
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

def plot_series(plot, show_plot_window):
  print(f"Ploting {plot['description']}")
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
    process_data(data=data, columns=plot_settings['columns_to_plot'], plot_type=plot_settings["plot_type"])

    ax1 = plt.subplot(111)
    #plt.tick_params('x', labelbottom=False)
    prepare_axis(ax=ax1, axis_settings=plot_settings["axis_settings"])

    plot_data(ax1, data,  plot_settings["x-axis"], plot_settings['columns_to_plot'])

    lines, labels = ax1.get_legend_handles_labels()

    plot_settings = plot.get('secondary_axis', {})
    if plot_settings.get("show", False):
      ax2 = ax1.twinx()
      prepare_axis(ax=ax2, axis_settings=plot_settings["axis_settings"])
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
  if show_plot_window:
    plt.show()

  plt.close(fig)


phi = (5**.5-1) / 2  # golden ratio


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic plot generator for line and scatter plots.")
    parser.add_argument("-v", "--version", action="version", version=f"{parser.prog} version {__version__}")
    parser.add_argument("plotfile", help="One or more yaml configurations to plot.")
    parser.add_argument('--silent', action='store_true', help="Do not show the plot when set.")

    return parser


if __name__ == "__main__":
  parser = init_argparse()
  args = parser.parse_args()
  plot_files = glob.glob(args.plotfile)
  for file_path in plot_files:
      module_name = os.path.splitext(os.path.basename(file_path))[0]
      spec = importlib.util.spec_from_file_location(module_name, file_path)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      try:
        plot_series(plot=module.plot, show_plot_window=not args.silent)
      except FileNotFoundError as exc:
          print(f"  Data file not found. Cannot plot graph: {exc}")
