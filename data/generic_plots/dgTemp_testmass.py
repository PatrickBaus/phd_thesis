import numpy as np
import pandas as pd
import seaborn as sns


def convertResistanceToTemperature(values):
    # Constants for Amphenol DC95 (Material Type 10kY)
    a = 3.3540153 * 10**-3
    b = 2.7867185 * 10**-4
    c = 4.0006637 * 10**-6
    d = 1.5575628 * 10**-7
    rt25 = 10 * 10**3

    return (
        1 / (a + b * np.log(values / rt25) + c * np.log(values / rt25) ** 2 + d * np.log(values / rt25) ** 3) - 273.15
    )


colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "DgTemp Testmass",
    "show": True,
    "output_file": {"fname": "../images/dgTemp_testmass.pgf"},
    "crop": {
        "crop_index": "date",
        "crop": ["2018-10-24 16:00:00", "2018-10-25 04:00:00"],
    },
    "legend_position": "upper right",
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Temperature deviation in \unit{\K}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -6,
            "x_scale": "time",
            "y_scale": "lin",
            # "limits_y": [22.75, 23.25],
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "temperature_mean": {
                "label": r"Temperature (Fluke 5611T-P)",
                "color": colors[0],
            },
        },
    },
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Ambient temperature in  \unit{\celsius}",
            "invert_x": False,
            "invert_y": False,
            # "fixed_order": -6,
            "x_scale": "lin",
            "y_scale": "lin",
            "show_grid": False,
            # "limits_y": [30.08,30.15],
        },
        "x-axis": "date",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "temperature": {
                "label": r"Ambient temperature",
                "color": colors[1],
            },
        },
    },
    "files": [
        {
            "filename": "HP3458A_GPIB_Read_2018-10-24_14:16:24+00:00.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "date",
                    1: "voltage",
                },
                "scaling": {
                    "temperature_mass": lambda x: convertResistanceToTemperature(x["voltage"] / (50 * 10**-6)),
                    "temperature_mean": lambda x: x["temperature_mass"] - x["temperature_mass"].mean() - 40e-6,
                    "date": lambda data: pd.to_datetime(data.date, utc=True),
                },
            },
        },
        {
            "filename": "fluke1524_2018-10-23_16:38:38+00:00.csv",  # Fixed direction. Going up now
            "show": True,
            "parser": "fluke1524",
            "options": {
                "sensor_id": 2,  # Fluke Sensor 1 = Board Temp, Sensor 2 = Ambient
                "scaling": {
                    "temperature": lambda x: x["temperature"] - 273.15,
                },
            },
        },
    ],
}
