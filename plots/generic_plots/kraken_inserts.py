import pandas as pd
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": r"Labkraken",
    "show": True,
    "output_file": {"fname": "../images/kraken_inserts.pgf"},
    # 'crop': ['2017-12-03 00:00:00', '2017-12-04 00:00:00'],
    # "legend_position": "lower center",
    "crop_secondary_to_primary": True,
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Number of daily database inserts",
            "invert_x": False,
            "invert_y": False,
            # "fixed_order": -6,
            "x_scale": "time",
            "y_scale": "log",
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "counts": {
                "label": r"\unit{inserts \per \day}",
                "color": colors[0],
            },
        },
    },
    "files": [
        {
            "filename": "kraken_database_counts.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "date",
                    1: "counts",
                },
                "scaling": {
                    "date": lambda data: pd.to_datetime(data.date, utc=True),
                },
            },
        },
    ],
}
