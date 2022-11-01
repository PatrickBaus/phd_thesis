#!/usr/bin/env python
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.legend
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from scipy.stats.distributions import t

from si_prefix import si_format

pd.plotting.register_matplotlib_converters()

from file_parser import parse_file

# Use these setting for the PhD thesis
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
})
plt.style.use('tableau-colorblind10')
# end of settings

def exponential_decay(t, a, tau, t0, offset):
    # Create a step function that is 0 for time values < t0 and 1 for the rest
    S = [0 if value < t0 else 1 for value in t]

    model = a * np.exp(-(t-t0)/tau*S) + offset
    return model

def fit_exponential_decay(x_data, y_data, initial_theta):
    t = x_data.values
    initial_t0 = 0
    initial_offset = min(y_data.values)
    initial_start = max(y_data.values) - initial_offset
    return curve_fit(
        exponential_decay,
        t, y_data.values,
        p0=[initial_start, 1e3, initial_t0, initial_offset],
        #bounds=([-np.inf, -np.inf, 0, 0, 0], np.inf),
    )


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

def crop_data(data, zoom_date=None, crop_secondary=None):
    if zoom_date is not None:
        index_to_drop = data[(data.date < zoom_date[0]) | (data.date > zoom_date[1])].index
        data.drop(index_to_drop , inplace=True)

    #y = 1/(0.000858614 + 0.000259555 * np.log(y) + 1.35034*10**-7 * np.log(y)**3)
    print(f"    Begin date: {data.date.iloc[0].tz_convert('Europe/Berlin')}")
    print(f"    End date:   {data.date.iloc[-1].tz_convert('Europe/Berlin')} (+{(data.date.iloc[-1]-data.date.iloc[0]).total_seconds()/3600:.1f} h)")

def prepare_axis(ax, label, color_map=None, fixed_order=None):
  if fixed_order is not None:
    ax.yaxis.set_major_formatter(FixedOrderFormatter(fixed_order, useOffset=True))
  else:
    ax.yaxis.get_major_formatter().set_useOffset(False)

  if color_map is not None:
    ax.set_prop_cycle('color', color_map)

  ax.set_ylabel(label)

def plot_data(ax, data, column, labels, linewidth):
  ax.plot(data.date, data[column], marker="", label=labels[column], alpha=0.7, linewidth=linewidth)

def plot_series(plot):
  # Load the data to be plotted
  plot_files = (plot_file for plot_file in plot['files'] if plot_file.get('show', True))
  data = pd.concat(
    (load_data(plot_file)[0] for plot_file in plot_files),
    sort=True
  )
  # Drop non-complete rows
  if plot.get('secondary_axis', {}).get('show', True):
    data.dropna(subset=[plot['primary_axis']['columns_to_plot'][0], plot['secondary_axis']['columns_to_plot'][0]], inplace=True)  # It is ok to drop a few values
  data.reset_index(drop=True, inplace=True)

  # If we have something to plot, proceed
  if not data.empty:
    crop_data(data, zoom_date=plot.get('zoom'), crop_secondary=plot.get('crop_secondary_to_primary'))

    plot_settings = plot['primary_axis']

    ax1 = plt.subplot(111)
    prepare_axis(ax=ax1, fixed_order=plot_settings['axis_fixed_order'], label=plot_settings['label'], color_map=plt.cm.tab10.colors)

    # Reset the time axis to 0 at the unit step
    step_index = data[data[plot_settings['columns_to_plot'][0]] != data[plot_settings['columns_to_plot'][0]].shift()].index[1]
    data.date = (data.date - data.date[step_index]).dt.total_seconds()

    plot_data(ax1, data, plot_settings['columns_to_plot'][0], plot_settings['labels'], linewidth=0.5)
    lines, labels = ax1.get_legend_handles_labels()

    if plot.get('secondary_axis', {}).get('show', True):
      ax2 = ax1.twinx()
      plot_settings2 = plot['secondary_axis']
      prepare_axis(ax=ax2, fixed_order=plot_settings2.get('axis_fixed_order'), label=plot_settings2['label'], color_map=['firebrick', 'green',])
      plot_data(ax2, data, plot_settings2['columns_to_plot'][0], plot_settings2['labels'], linewidth=0.5)
      lines2, labels2 = ax2.get_legend_handles_labels()
      lines += lines2
      labels += labels2

      # We need a initial guess of the dead time
      # We assume it is 0
      params, pcov = fit_exponential_decay(pd.to_numeric(data.date), data[plot_settings2['columns_to_plot'][0]], initial_theta=0)
      # Calculate the uncertainty (95%) of the constants
      alpha = 1 - 0.954499736103642 # 95% confidence interval = 100*(1-alpha)
      n = len(data)    # number of data points
      n_params = len(params) # number of constants
      dof = max(0, n - n_params) # number of degrees of freedom
      # student-t value for the dof and confidence level
      tval = t.ppf(1.0-alpha/2., dof)
      sigma_squared = np.diag(pcov)
      output_step = max(data[plot_settings['columns_to_plot'][0]]) - min(data[plot_settings['columns_to_plot'][0]])
      system_step = params[0]
      K = system_step/output_step
      T = params[1]
      tau = params[2]
      print(f"Normalized step K: {system_step} K / {output_step} bit ({K} ± {sigma_squared[0]**0.5*tval/output_step}) K/bit")
      print(f"Decay time T ({T} ± {sigma_squared[1]**0.5*tval}) s")
      print(f"Dead time τ: ({tau} ± {sigma_squared[2]**0.5*tval}) s")
      print(f"PI parameters (Ziegler–Nichols): Kp={0.9*T/(K*tau)} bit/K, Ki={0.3*T/(K*tau**2)} bit/(Ks)")
      print(f"PI parameters (SIMC): Kp={T/(2*K*tau)} bit/K, Ki={1/(2*K*tau)} bit/(Ks)")
      print(f"PI parameters (Advanced PID control): Kp={(0.15*tau+0.35*T)/(K*tau)} bit/K, Ki={(0.46*tau+0.02*T)/(K*tau**2)} bit/(Ks)")
      print(f"PI parameters (APQ): Kp={(0.15*tau+0.35*T)/(K*tau)/6:.2f} bit/K, Ki={(0.46*tau+0.02*T)/(K*tau**2)/4:.2f} bit/(Ks)")
      print(f"PID setpoint: {system_step+params[3]:.1f} °C")
      data["fit"] = exponential_decay(pd.to_numeric(data.date.values), *params)
      plot_data(ax2, data, "fit", plot_settings2['labels'], linewidth=2)

    #ax1.set_ylabel(plot_settings['label'])
    #ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
    #ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax1.set_xlabel("Time in seconds")

    plt.legend(lines, labels, loc='upper right')

  fig = plt.gcf()
#  fig.set_size_inches(11.69,8.27)   # A4 in inch
#  fig.set_size_inches(128/25.4 * 2.7 * 0.8, 96/25.4 * 1.5 * 0.8)  # Latex Beamer size 128 mm by 96 mm
  fig.set_size_inches(418.25555 / 72.27 * 0.9, 418.25555 / 72.27*(5**.5 - 1) / 2 * 0.9)  # TU thesis
  if plot.get('title') is not None:
    plt.suptitle(plot['title'], fontsize=16)

  plt.tight_layout()
  if plot.get('title') is not None:
    plt.subplots_adjust(top=0.88)
  plt.show()

if __name__ == "__main__":
  plots = [
    {
      'title': "K01 Server Room",
      'title': None,
      'show': False,
      #'zoom': ['2022-09-13 04:34:00', '2025-06-26 00:00:00'],  # Only exponential decay
      'zoom': ['2022-09-13 03:30:00', '2025-09-13 04:30:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663079073167.csv' ,  # Server Room, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_2',
          'options': {
            'scaling': {
              'temperature': lambda x : x['temperature'] - 273.15,
            },
          },
        },
      ],
    },
    {
      'title': "011 Neon Lab (Front)",
      'title': None,
      'show': False,
      #'zoom': ['2022-09-13 03:30:00', '2025-09-13 04:30:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663476153397.csv' ,  # Neon front, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_2',
          'options': {
            'scaling': {
              'temperature': lambda x : x['temperature'] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "011 Neon Lab (Back)",
      'title': None,
      'show': False,
      'zoom': ['2022-09-13 03:30:00', '2022-09-22 07:20:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature_labnode'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature_labnode": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663917584771.csv' ,  # Neon back, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "012 Laser Lab (Front)",
      'title': None,
      'show': False,
      #'zoom': ['2022-09-22 18:30:00', '2022-09-22 20:00:00'],  # Failed attempt
      #'zoom': ['2022-09-22 18:30:00', '2022-09-22 20:00:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature_labnode'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature_labnode": "Temperature (Labnode)",
          "temperature_room": "Temperature (Room)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663924972026.csv' ,  # Failed attempt, moved controller
          'show': False,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
            },
          },
        },
        {
          'filename': 'data-1663918677781.csv' ,  # Laser lab front, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "012 Laser Lab (Back)",
      'title': None,
      'show': False,
      'zoom': ['2022-09-21 02:30:00', '2022-09-21 04:20:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature_labnode'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature_labnode": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663919213658.csv' ,  # Laser lab back, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "015 ATOMICS Lab (Front)",
      'title': None,
      'show': False,
      'zoom': ['2021-09-13 03:30:00', '2022-09-21 02:40:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature_labnode'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature_labnode": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663737298096.csv' ,  # ATOMICS front, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "015 ATOMICS Lab (Back)",
      'title': None,
      'show': False,
      'zoom': ['2021-09-13 03:30:00', '2022-09-22 21:10:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature_labnode'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature_labnode": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1663913960965.csv',  # ATOMICS back, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
            },
          },
        },
      ],
    },
    {
      'title': "016 Ti:Sa",
      'title': None,
      'show': True,
      #'zoom': ['2021-09-13 03:30:00', '2022-09-22 21:10:00'],  # This range start reasonably flat
      'crop_secondary_to_primary': False,
      'primary_axis': {
        'label': "DAC outpt in bit",
        'axis_fixed_order': 0,
        'columns_to_plot': ["output",],
        'filter': None,
        # filter_savgol(window_length=101, polyorder=3),
        'labels': {
          "output": "DAC output",
        },
        'options': {
          'show_filtered_only': False,
        }
      },
      'secondary_axis': {
        'show': True,
        "label": "Temperature in °C",
        'plot_type': 'absolute',
        "unit": "°C",
        'columns_to_plot': ['temperature_labnode'],
#        'filter': filter_savgol(window_length=151, polyorder=3),
        "labels": {
          "temperature_labnode": "Temperature (Labnode)",
          "fit": "Fit",
        },
#        'axis_fixed_order': -6,
        "options": {
          "invert_axis": False,
        }
      },
      'files': [
        {
          'filename': 'data-1664095658934.csv',  # ATOMICS back, 10k/10k resistors
          'show': True,
          'parser': 'timescale_db_3',
          'options': {
            'scaling': {
              'temperature_labnode': lambda x : x['temperature_labnode'] -273.15,
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
