import numpy as np
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "DgDrive Output Impedance",
    "show": True,
    "output_file": {"fname": "../images/dgDrive_output_impedance_dc.pgf"},
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "legend_position": "best",
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Time in \unit{\s}",
            "y_label": r"Output current deviation in \unit{\A}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -9,
            "x_scale": "lin",
            "y_scale": "lin",
        },
        "x-axis": "time",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "value": {
                "label": None,
                "color": colors[0],
            },
            "lower": {
                "label": None,
                "color": colors[1],
            },
            "upper": {
                "label": None,
                "color": colors[1],
            },
        },
    },
    "files": [
        {
            "filename": "DgDrive_output_impedance_3M3_10PLC_AZ.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "time",
                    1: "value",
                },
                "skiprows": 2,
                "scaling": {
                    "time": lambda x: x["time"] / 2.5,  #  NPLC=10 + autozero
                    "lower": lambda x: np.where(x["time"] <= 30.5, np.mean(x["value"][x["time"] <= 30.5]), np.nan),
                    "upper": lambda x: np.where(x["time"] > 30.5, np.mean(x["value"][x["time"] > 30.5]), np.nan),
                },
            },
        },
    ],
}
