import pandas as pd
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "Room temperature Neon",
    "show": True,
    "output_file": {"fname": "../images/temperature_011_2016.pgf"},
    #'crop': ['2018-10-24 16:00:00', '2018-10-25 04:00:00'],
    "legend_position": "upper right",
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Temperature in \unit{\celsius}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": 0,
            "x_scale": "time",
            "y_scale": "lin",
            # "limits_y": [22.75, 23.25],
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "temperature": {
                "label": r"Room temperature",
                "color": colors[0],
            },
        },
    },
    "files": [
        {
            "filename": "011_neon_temperature_2016-11 00-00-00-26_2016-11-27 00-00-00.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "date",
                    1: "temperature",
                },
                "scaling": {
                    "date": lambda data: pd.to_datetime(data.date, utc=True),
                    "temperature": lambda data: data["temperature"] - 273.15,
                },
            },
        },
    ],
}
