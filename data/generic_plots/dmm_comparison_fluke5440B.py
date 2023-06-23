import pandas as pd
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "DMM comparison vs. Fluke 5440B",
    "show": True,
    "output_file": {"fname": "../images/dmm_comparison_fluke5440B.pgf"},
    #'crop': [0,31e-3],
    "legend_position": "lower center",
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Voltage deviation in \unit{\V}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -6,
            "x_scale": "time",
            "y_scale": "lin",
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
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
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Ambient temperature in \unit{\celsius}",
            "invert_x": False,
            "invert_y": False,
            # "fixed_order": -6,
            "x_scale": "lin",
            "y_scale": "lin",
            "show_grid": False,
            "limits_y": [22.4, 23.4],
        },
        "x-axis": "date",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "temperature": {
                "label": "Ambient temperature",
                "color": colors[3],
            },
        },
        "filter": None,  # filter_savgol(window_length=101, polyorder=3),
    },
    "files": [
        {
            "filename": "data-1664633967407.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "date",
                    1: "KS34470A",
                    2: "HP3458A",
                    3: "K2002",
                    4: "DMM6500",
                    5: "temperature",
                },
                "scaling": {
                    "date": lambda data: pd.to_datetime(data.date, utc=True),
                    "KS34470A": lambda data: data["KS34470A"] - data["KS34470A"].mean() + 2 * 10**-5,
                    "HP3458A": lambda data: data["HP3458A"] - data["HP3458A"].mean() + 10**-5,
                    "K2002": lambda data: data["K2002"] - data["K2002"].mean(),
                    "DMM6500": lambda data: data["DMM6500"] - data["DMM6500"].mean() - 10**-5,
                },
            },
        },
    ],
}
