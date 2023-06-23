import pandas as pd
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": r"LabNode controller",
    "show": True,
    "output_file": {"fname": "../images/labnode_performance.pgf"},
    # 'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
    # "legend_position": "lower center",
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "crop_secondary_to_primary": True,
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Temperature in \unit{\celsius}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": None,
            "x_scale": "time",
            "y_scale": "lin",
            "limits_y": [22.75, 23.25],
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "temperature_in_loop": {
                "label": r"Temperature (in loop)",
                "color": colors[0],
            },
        },
    },
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Temperature in \unit{\celsius}",
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
            "temperature_out_of_loop": {
                "label": r"Temperature (rack)",
                "color": colors[1],
            },
        },
    },
    "files": [
        {
            "filename": "Neon Lab - 011-data-as-joinbyfield-2023-05-06 20_14_00.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "date",
                    1: "temperature_in_loop",
                    7: "temperature_out_of_loop",
                },
                "scaling": {
                    "date": lambda data: pd.to_datetime(data.date, utc=True),
                    "temperature_in_loop": lambda data: data["temperature_in_loop"]
                    .str.removesuffix(" °C")
                    .astype(float),
                    "temperature_out_of_loop": lambda data: data["temperature_out_of_loop"]
                    .str.removesuffix(" °C")
                    .astype(float),
                },
            },
        },
    ],
}
