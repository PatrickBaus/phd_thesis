import seaborn as sns
import numpy as np

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio


def calculate_saturation_current(vds, kappa, ld):
    return 0.5 * kappa * vds**2 * (1 + ld * vds)


plot = {
    "description": "MOSFET Id LTSpice example",
    "show": True,
    "output_file": {"fname": "../images/ltspice_mosfet_drain-current_example.pgf"},
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Drain-source voltage $V_{DS}$ in \unit{\V}",
            "y_label": r"Drain Current $I_{D}$ in \unit{\A}",
            "invert_x": True,
            "invert_y": True,
            "fixed_order": -3,
            "y_scale": "linear",
        },
        "x-axis": "vds",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "Vgs0.2": {
                "label": "$V_{GS} + V_{th} = \qty{-0.2}{\V}$",
                "color": colors[0],
            },
            "Vgs0.4": {
                "label": "$V_{GS} + V_{th} = \qty{-0.4}{\V}$",
                "color": colors[1],
            },
            "Vgs0.6": {
                "label": "$V_{GS} + V_{th} = \qty{-0.6}{\V}$",
                "color": colors[2],
            },
            "Vgs0.8": {
                "label": "$V_{GS} + V_{th} = \qty{-0.8}{\V}$",
                "color": colors[3],
            },
            "isat": {
                "label": "$I_{sat}$",
                "color": colors[4],
                "linestyle": "--",
            },
        },
    },
    "files": [
        {
            "filename": "mosfet_id.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    i: val
                    for i, val in enumerate(
                        [
                            "Vsd",
                        ]
                        + [f"Vgs{val:.1f}" for val in np.arange(0.2, 0.9, 0.2)]
                    )
                },
                "scaling": {
                    "vds": lambda x: -x["Vsd"],
                    "isat": lambda x: calculate_saturation_current(x["vds"][x["vds"] >= -0.81], -0.813, -4 / 1000),
                },
            },
        },
    ],
}
