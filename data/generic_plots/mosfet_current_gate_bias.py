import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "IRF9610 MOSFET simulation",
    "show": True,
    "output_file": {"fname": "../images/mosfet_current_gate_bias.pgf"},
    "crop_secondary_to_primary": True,
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Drain-Source Voltage $V_{DS}$ in \unit{\V}",
            "y_label": r"Drain Current $I_D$ in \unit{\A}",
            "invert_x": True,
            "invert_y": True,
            "fixed_order": -3,
            "y_scale": "linear",
        },
        "x-axis": "vds",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "Vgs3.5": {
                "label": "$V_{GS} = \qty{-3.5}{\V}$",
                "color": colors[0],
            },
            "Vgs4.0": {
                "label": "$V_{GS} = \qty{-4}{\V}$",
                "color": colors[1],
            },
            "Vgs4.5": {
                "label": "$V_{GS} = \qty{-4.5}{\V}$",
                "color": colors[2],
            },
            "Vgs5": {
                "label": "$V_{GS} = \qty{-5}{\V}$",
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
            "filename": "mosfet_current_source.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {0: "vsd", 1: "Vgs3.5", 2: "Vgs4.0", 3: "Vgs4.5", 4: "Vgs5"},
                "scaling": {
                    "vds": lambda x: -x["vsd"],
                    #'isat': lambda x : calculate_saturation_current(x["vds"][x["vds"]>=-0.81], -0.813, -4/1000),
                },
            },
        },
    ],
}
