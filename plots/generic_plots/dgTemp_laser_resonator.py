import pandas as pd
import seaborn as sns
import numpy as np


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
    "description": "DgTemp Performance",
    "show": True,
    "output_file": {"fname": "../images/dgTemp_laser_resonator.pgf"},
    #'crop': ['2018-10-17 10:00:00', '2020-10-16 06:00:00'],   # Air drafts (outer silicone)
    "crop": {
        "crop_index": "date",
        "crop": (
            "2018-10-15 12:00:00",
            "2020-10-16 06:00:00",
        ),
    },  # Air drafts (outer silicone)
    "legend_position": "upper right",
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Temperature deviation in \unit{\K}",
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
            "temperature": {
                "label": r"Laser resonator temperature",
                "color": colors[0],
            },
            "setpoint": {
                "label": r"Setpoint \qty{21.6893}{\celsius}",
                "color": colors[1],
                "linestyle": "dashed",
            },
        },
    },
    "files": [
        {
            "filename": "ADC_Serial_Read_2018-10-17_07:54:38+00:00.csv",
            "filename": "ADC_Serial_Read_2018-10-15_10:25:38+00:00.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "skiprows": 7,
                "columns": {
                    1: "date",
                    2: "adc_code",
                },
                "scaling": {
                    "resistance": lambda x: x["adc_code"] / (2**31 - 1) * 4.096 / (50 * 10**-6),
                    "temperature": lambda x: convertResistanceToTemperature(x["resistance"])
                    - convertResistanceToTemperature(300000000 * 4.096 / (2**31 - 1) / (50 * 10**-6)),
                    "setpoint": lambda x: np.zeros(len(x)),
                    "date": lambda data: pd.to_datetime(data.date, utc=True, format="ISO8601"),
                },
            },
        },
        {
            "filename": "sensorData_2019-06-30 22_00_00_2019-07-18 06_34_00.csv",
            "show": False,
            "parser": "smi",
            "options": {
                "sensor_id": 9,
                "label": "humidity",
                "scaling": {},
            },
        },
    ],
}
