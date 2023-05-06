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
from scipy import stats
from statsmodels.formula.api import ols
from matplotlib.colors import ListedColormap
import lttb

pd.plotting.register_matplotlib_converters()

from file_parser import parse_file

colors = sns.color_palette("colorblind")
cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

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

  if axis_settings.get("show_grid", True):
    ax.grid(True, which="minor", ls="-", color='0.85')
    ax.grid(True, which="major", ls="-", color='0.45')
  #else:
  #  ax.grid(False, which="both")

  ax.set_ylabel(axis_settings["y_label"])
  if axis_settings.get("x_label") is not None:
    ax.set_xlabel(axis_settings["x_label"])

  if color_map is not None:
    ax.set_prop_cycle('color', color_map)

def fit_data(data, x_axis, y_axis):
    model = ols(f'{y_axis} ~ {x_axis}', data).fit()

    # Calculate uncertainty for 1 sigma from the standard error
    uncertainty = model.bse * stats.t.interval(0.997300203936740, len(data)-1)[1] # 1 sigma (1-alpha = 0.682689492137086) = 68%, 2 sigma = 0.954499736103642, 3 sigma = 0.997300203936740, etc

    return {"intercept":  model.params.Intercept,
            "slope": model.params[x_axis],
            "uncertainty": uncertainty[x_axis],
           }

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

def plot_data(ax, data, x_axis, column_settings):
  for column, settings in column_settings.items():
      if column in data:
          if "cmap" in settings:
            data_points = max(len(data)//2000,1)
            downsampled_data = data.iloc[::data_points]
            print(f"Scatter data downsampled to {len(downsampled_data)} points.")
            ax.scatter(
              downsampled_data[x_axis],
              downsampled_data[column],
              alpha=0.7,
              c=downsampled_data.date,
              **settings
            )
          else:
            x_data, y_data = downsample_data(data[x_axis], data[column])
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

  # If we have something to plot, proceed
  if not data.empty:
    crop_data(data, crop_index="date", crop=plot.get('crop'))

    plot_settings = plot['primary_axis']
    process_data(data=data, columns=plot_settings['columns_to_plot'], plot_type=plot_settings.get('plot_type','absolute'))

    if True:
      ax1 = plt.subplot(211)
      #plt.tick_params('x', labelbottom=False)
      prepare_axis(
          ax=ax1,
          axis_settings=plot_settings["axis_settings"],
          color_map=plt.cm.tab10.colors
        )

      x_axis = plot_settings["x-axis"]
      plot_data(ax1, data, x_axis=x_axis, column_settings=plot_settings['columns_to_plot'])

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

      plt.legend(lines, labels, loc=plot.get("legend_position", "upper left"))

    if True:
      ax3 = plt.subplot(212)
      plot_settings = plot['xy_plot']
      x_axis = plot_settings["x-axis"]
      y_axis = plot_settings["y-axis"]
      fit = fit_data(data, x_axis, y_axis)
      data["fit"] = fit["slope"] * data[x_axis] + fit["intercept"]
      prepare_axis(
          ax=ax3,
          axis_settings=plot_settings["axis_settings"],
          color_map=plt.cm.tab10.colors
      )
      plot_data(ax3, data, x_axis=x_axis, column_settings=plot_settings['columns_to_plot'])
      lines2, labels2 = ax3.get_legend_handles_labels()

      ax3.legend(lines2, labels2, loc=plot_settings.get("legend_position", "upper left"))
      ax3.annotate(
          f"Tempco: ({fit['slope']:.3e} ± {fit['uncertainty']:.2e}) A/K",
          xy=(0.9,0.1),
          xycoords='axes fraction',
          xytext=(-5, 0), textcoords='offset points', ha='right',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
      )

  fig = plt.gcf()
#  fig.set_size_inches(11.69,8.27)   # A4 in inch
#  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
  phi = (5**.5-1) / 2  # golden ratio
  fig.set_size_inches(441.01773 / 72.27 * 0.9, 441.01773 / 72.27 * 0.9 * phi * 2)  # y * 2 for 2 plots
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
      'title': 'Leakage Current Nichicon UKL',
      'title': None,
      'show': False,
      'crop': ['2019-12-28 02:00:00', '2019-12-28 18:00:00'],
      "output_file": {
          "fname": "../images/leakage_current_ukl.pgf",
      },
      "legend_position": "best",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Leakage current in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          #"y_scale": "lin",
          "x_scale": "time",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "value": {
                "label": "Leakage current",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        "show": True,
        "axis_settings": {
          "show_grid": False,
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": 9,
          #"y_scale": "lin",
          "x_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Temperature",
                "color": colors[3],
            },
        },
      },
      "xy_plot": {
          "legend_position": "upper left",
          'x-axis': "temperature",
          "y-axis": "value",
          'columns_to_plot': {
              "value": {
                  "label": "Leakage current",
                  "s": 1,  # point size
                  "cmap": cmap,
              },
              "fit": {
                  "label": "Regression",
                  "color": colors[3],
              },
          },
          "axis_settings": {
              'x_label': r"Temperature in \unit{\celsius}",
              'y_label': r"Leakage current in \unit{\A}",
              "invert_x": False,
              "invert_y": False,
              "fixed_order": -9,
              #"y_scale": "lin",
              "x_scale": "lin",
        },

      },
      'files': [
        {
          'filename': 'HP3458A_UKL_330uF_leakage_2019-12-27_11:25:34+00:00.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                1: "value",
                2: "temperature",
            },
            "scaling": {
                "value": lambda x: x.value / 2.192e6,  # divide voltage by 2.192e6
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
      ],
    },
    {
      'title': 'Leakage Current WIMA MKS4',
      'title': None,
      'show': False,
      'crop': ['2020-03-19 20:00:00', '2023-03-20 10:11:49'],
      "output_file": {
          "fname": "../images/leakage_current_mks4.pgf",
      },
      "legend_position": "upper right",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Leakage current in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          #"y_scale": "lin",
          "x_scale": "time",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "value": {
                "label": "Leakage current",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        "show": True,
        "axis_settings": {
          "show_grid": False,
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": 9,
          #"y_scale": "lin",
          "x_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Temperature",
                "color": colors[3],
            },
        },
      },
      "xy_plot": {
          "legend_position": "upper left",
          'x-axis': "temperature",
          "y-axis": "value",
          'columns_to_plot': {
              "value": {
                  "label": "Leakage current",
                  "s": 1,  # point size
                  "cmap": cmap,
              },
              "fit": {
                  "label": "Regression",
                  "color": colors[3],
              },
          },
          "axis_settings": {
              'x_label': r"Temperature in \unit{\celsius}",
              'y_label': r"Leakage current in \unit{\A}",
              "invert_x": False,
              "invert_y": False,
              "fixed_order": -9,
              #"y_scale": "lin",
              "x_scale": "lin",
        },

      },
      'files': [
        {
          'filename': 'HP3458A_MKS4_150uF_50V_leakage_2020-03-19_13:51:19+00:00.csv',
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "columns": {
                0: "date",
                2: "value",
                3: "temperature",
            },
            "scaling": {
                "value": lambda x: x.value / 2.192e6,  # divide voltage by 2.192e6
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
      ],
    },
    {
      'title': r"DgDrive-500-LN v2.3.1 (\#14, \qty{50}{\mA}) Tempco test",
      'title': None,
      'show': True,
      'crop': ['2021-03-15 6:00:00', '2022-03-03 17:00:00'],
      "output_file": {
          "fname": "../images/dgDrive_tempco_50mA.pgf",
      },
      "legend_position": "upper right",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          #"y_scale": "lin",
          "x_scale": "time",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "value": {
                "label": "Leakage current",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        "show": True,
        "axis_settings": {
          "show_grid": False,
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": 9,
          #"y_scale": "lin",
          "x_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Temperature",
                "color": colors[3],
            },
        },
      },
      "xy_plot": {
          "legend_position": "upper left",
          'x-axis': "temperature",
          "y-axis": "value",
          'columns_to_plot': {
              "value": {
                  "label": "Current deviation",
                  "s": 1,  # point size
                  "cmap": cmap,
              },
              "fit": {
                  "label": "Regression",
                  "color": colors[3],
              },
          },
          "axis_settings": {
              'x_label': r"Temperature in \unit{\celsius}",
              'y_label': r"Current deviation in \unit{\A}",
              "invert_x": False,
              "invert_y": False,
              "fixed_order": -9,
              #"y_scale": "lin",
              "x_scale": "lin",
        },

      },
      'files': [
        {
          'filename': "DgDrive-2-3-1_Tempco_2021-03-16_15:59:05+00:00.csv",   # Fixed direction. Going up now
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "skiprows": 7,
            "columns": {
                0: "date",
                1: "value",
                3: "temperature",
            },
            "scaling": {
                "value": lambda x : (x["value"] - x["value"].mean()) / 100,  # in A and relative coordinates
                "date": lambda data: pd.to_datetime(data.date, utc=True),
            },
          },
        },
      ],
    },
{
      'title': r"DgDrive-500-LN v2.3.1 (\#14, \qty{510}{\mA}) Tempco test",
      'title': None,
      'show': True,
      'crop': ['2021-03-27 01:00:00', '2022-03-03 17:00:00'],
      "output_file": {
          "fname": "../images/dgDrive_tempco_510mA.pgf",
      },
      "legend_position": "upper right",
      'primary_axis': {
        "axis_settings": {
          'x_label': r"Time (UTC)",
          'y_label': r"Current deviation in \unit{\A}",
          "invert_x": False,
          "invert_y": False,
          "fixed_order": -9,
          #"y_scale": "lin",
          "x_scale": "time",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "value": {
                "label": "Leakage current",
                "color": colors[0],
            },
        },
      },
      'secondary_axis': {
        "show": True,
        "axis_settings": {
          "show_grid": False,
          'y_label': r"Temperature in \unit{\celsius}",
          "invert_x": False,
          "invert_y": False,
          #"fixed_order": 9,
          #"y_scale": "lin",
          "x_scale": "lin",
        },
        'x-axis': "date",
        'plot_type': 'absolute',  # absolute, relative, proportional
        'columns_to_plot': {
            "temperature": {
                "label": "Temperature",
                "color": colors[3],
            },
        },
      },
      "xy_plot": {
          "legend_position": "upper left",
          'x-axis': "temperature",
          "y-axis": "value",
          'columns_to_plot': {
              "value": {
                  "label": "Current deviation",
                  "s": 1,  # point size
                  "cmap": cmap,
              },
              "fit": {
                  "label": "Regression",
                  "color": colors[3],
              },
          },
          "axis_settings": {
              'x_label': r"Temperature in \unit{\celsius}",
              'y_label': r"Current deviation in \unit{\A}",
              "invert_x": False,
              "invert_y": False,
              "fixed_order": -9,
              #"y_scale": "lin",
              "x_scale": "lin",
        },

      },
      'files': [
        {
          'filename': "DgDrive-2-3-1_Tempco_2021-03-23_14:41:01+00:00.csv",   # Fixed direction. Going up now
          'show': True,
          'parser': 'ltspice_fets',
          'options': {
            "skiprows": 7,
            "columns": {
                0: "date",
                1: "value",
                6: "temperature",
            },
            "scaling": {
                "value": lambda x : (x["value"] - x["value"].mean()) / 10,  # in A and relative coordinates
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
