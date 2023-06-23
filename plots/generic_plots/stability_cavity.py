import pandas as pd
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": r" Stable Laser Systems VH 6020-4 (\qty{840}{\nm}) temperature over \qty{24}{\h}",
    "show": True,
    "output_file": {"fname": "../images/stability_cavity.pgf"},
    # 'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
    # "legend_position": "lower center",
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Temperature in \unit{\celsius}",
            "invert_x": False,
            "invert_y": False,
            # "fixed_order": -6,
            "x_scale": "time",
            "y_scale": "lin",
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "temperature": {
                "label": r"Temperature",
                "color": colors[0],
            },
        },
    },
    "files": [
        {
            "filename": "Temperature Cavity-data-2023-05-04.csv",
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
