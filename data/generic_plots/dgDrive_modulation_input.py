import numpy as np
import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "DgDrive Modulation Input",
    "show": True,
    "output_file": {"fname": "../images/dgDrive_modulation_input.pgf"},
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "crop": {
        "crop_index": "frequency",
        "crop": [1e2, 5e6],
    },
    "legend_position": "best",
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Frequency in \unit{\Hz}",
            "y_label": r"Amplitude in \unit{\decibel}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": None,
            "x_scale": "log",
            "y_scale": "lin",
        },
        "x-axis": "frequency",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "amplitude_normalized": {
                "label": r"Normalized amplitude",
                "color": colors[0],
            },
            "3dB": {
                "label": r"\qty{\pm 3}{\decibel}",
                "color": colors[1],
                "linestyle": (0, (5, 10)),  # loosely dashed
            },
            "-3dB": {
                "label": None,
                "color": colors[1],
                "linestyle": (0, (5, 10)),  # loosely dashed
            },
        },
    },
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            "y_label": r"Phase in \unit{\degree}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": None,
            "x_scale": "log",
            "y_scale": "lin",
            "show_grid": False,
        },
        "x-axis": "frequency",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "phase_corrected": {
                "label": r"Phase",
                "color": colors[2],
            },
        },
    },
    "files": [
        {
            "filename": "DgDrive_modulation_input.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    1: "frequency",
                    3: "amplitude",
                    4: "phase",
                    5: "reference",
                    6: "reference_phase",
                },
                "scaling": {
                    "amplitude_corrected": lambda x: x["amplitude"] - x["reference"],
                    "amplitude_normalized": lambda x: x["amplitude_corrected"]
                    - np.mean(x["amplitude_corrected"][(x["frequency"] >= 1e3) & (x["frequency"] <= 1e5)]),
                    "3dB": lambda x: np.repeat(
                        np.mean(x["amplitude_normalized"][(x["frequency"] >= 1e3) & (x["frequency"] <= 1e5)] + 3),
                        len(x),
                    ),
                    "-3dB": lambda x: np.repeat(
                        np.mean(x["amplitude_normalized"][(x["frequency"] >= 1e3) & (x["frequency"] <= 1e5)] - 3),
                        len(x),
                    ),
                    "phase_corrected": lambda x: x["phase"] - x["reference_phase"],
                },
            },
        },
    ],
}
