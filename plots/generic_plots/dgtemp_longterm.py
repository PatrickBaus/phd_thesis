import pandas as pd
import seaborn as sns


colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "DgTemp Longterm",
    "show": False,
    "output_file": {"fname": "../images/dgTemp_longterm.pgf"},
    "crop": {
        "crop_index": "date",
        "crop": (
            "2019-07-06 00:00:00",
            "2019-07-12 16:00:00",
        ),  # Drift, Humidity? -> Yes
    },
    "legend_position": "lower left",
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Resistance deviation in \unit{\ohm}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -3,
            "x_scale": "time",
            "y_scale": "lin",
            # "limits_y": [22.75, 23.25],
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "fluke1590": {
                "label": r"Fluke1590",
                "color": colors[4],
            },
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
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Humidity in \unit{\percent RH}",
            "invert_x": False,
            "invert_y": True,
            # "fixed_order": -6,
            "x_scale": "lin",
            "y_scale": "lin",
            "show_grid": False,
            # "limits_y": [30.08,30.15],
        },
        "x-axis": "date",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "humidity": {
                "label": r"Humdity",
                "color": colors[9],
            },
        },
    },
    "files": [
        {
            "filename": "../data/Rev2_INL_2019-07-02_08:08:20+00:00.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    1: "date",
                    2: "value_ext",
                    4: "value_int",
                },
                "scaling": {
                    "value_ext": lambda x: (x["value_ext"] - x["value_ext"].mean())
                    / (2**31 - 1)
                    * 4.096
                    / (50 * 10**-6)
                    - 14e-3,
                    "value_int": lambda x: (x["value_int"] - x["value_int"].mean())
                    / (2**31 - 1)
                    * 4.096
                    / (50 * 10**-6)
                    + 2e-3,
                    "date": lambda data: pd.to_datetime(data.date, utc=True, format="ISO8601"),
                },
            },
        },
        {
            "filename": "../data/fluke_1590_2023-06-05_05:22:31+00:00.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "date",
                    1: "fluke1590",
                },
                "scaling": {
                    "fluke1590": lambda data: (data["fluke1590"] - data["fluke1590"].mean()),
                    "date": lambda data: pd.to_datetime(data.date, utc=True)
                    - pd.to_datetime(data.date.iloc[0], utc=True)
                    + pd.Timestamp("2019-07-06 00:00:00", tz="UTC"),
                },
            },
        },
        {
            "filename": "../data/sensorData_2019-06-30 22_00_00_2019-07-18 06_34_00.csv",
            "show": True,
            "parser": "smi",
            "options": {
                "sensor_id": 9,
                "label": "humidity",
                "scaling": {},
            },
        },
    ],
}
