import pandas as pd
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": r"SMC11 stability over \qty{24}{\hour}",
    "show": True,
    "output_file": {"fname": "../images/stability_smc11.pgf"},
    "crop": {
        "crop_index": "date",
        "crop": ["2019-12-01 00:00:00", "2019-12-02 00:00:00"],
        "crop": [0, 24],
    },
    "legend_position": "upper right",
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time in \unit{\hour}",
            "y_label": r"Current deviation in \unit{\A}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -6,
            "x_scale": "timedelta",
            "y_scale": "lin",
            "limits_y": [-3e-6, 3e-6],
        },
        "x-axis": "date",
        "plot_type": "absolute",
        "columns_to_plot": {
            "smc11": {
                "label": r"SMC11",
                "color": colors[4],
            },
            "dgDrive": {
                "label": r"DgDrive-500-LN",
                "color": colors[3],
            },
        },
    },
    "secondary_axis": {
        "show": False,
        "axis_settings": {
            "x_label": r"Time (UTC)",
            "y_label": r"Ambient temperature in \unit{\celsius}",
            "invert_x": False,
            "invert_y": False,
            # "fixed_order": -6,
            "x_scale": "lin",
            "y_scale": "lin",
            "show_grid": False,
        },
        "x-axis": "date",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "temperature": {
                "label": "Ambient temperature",
                "color": colors[0],
            },
        },
    },
    "files": [
        {
            "filename": "SMC11_50mA+10R_60h_2.csv",
            "show": True,
            "parser": "34470A",
            "options": {
                "value_name": "smc11",
                "scaling": {
                    "date": lambda data: (
                        data["date"] - pd.Timestamp("2019-12-01 00:00:00", tz="UTC")
                    ).dt.total_seconds()
                    / 3600,
                    "smc11": lambda data: (data["smc11"] - data["smc11"].mean()) / 10,
                },
            },
        },
        {
            "filename": "DgDrive-1-2-1_50mA_FilmCap+RefNo3_10R_34470A_60h_8.csv",
            "show": True,
            "parser": "34470A",
            "options": {
                "value_name": "dgDrive",
                "scaling": {
                    "date": lambda data: (
                        data["date"] - pd.Timestamp("2019-11-23 12:00:00", tz="UTC")
                    ).dt.total_seconds()
                    / 3600,
                    "dgDrive": lambda data: (
                        data["dgDrive"] - data["dgDrive"][(data["date"] >= 0) & (data["date"] <= 24)].mean()
                    )
                    / 10
                    - 2e-6,
                },
            },
        },
    ],
}
