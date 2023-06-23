import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": r"DgDrive input filter simulation",
    "show": True,
    "output_file": {"fname": "../images/input_filter_dgdrive.pgf"},
    "legend_position": "upper right",
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Frequency in \unit{\Hz}",
            "y_label": r"Magnitude in \unit{\V \per \V}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -3,
            "x_scale": "log",
            "y_scale": "log",
        },
        "x-axis": "freq",
        "plot_type": "absolute",
        "columns_to_plot": {
            "lc_filter": {
                "label": "Mag. LC only",
                "color": colors[0],
            },
            "cap_mult": {
                "label": "Mag. LC + C Mult.",
                "color": colors[1],
            },
        },
    },
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            #'x_label': r"Time (UTC)",
            "y_label": r"Impedance in \unit{\ohm}",
            "invert_x": False,
            "invert_y": False,
            # "fixed_order": -6,
            "x_scale": "log",
            "y_scale": "lin",
            "show_grid": False,
            # "limits_y": [22.4,23.4],
        },
        "x-axis": "freq",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "z_lc": {
                "label": r"$Z_{out}$ LC filter",
                "color": colors[2],
                "linestyle": "--",
            },
            "z_cap_mult": {
                "label": r"$Z_{out}$ C Mult.",
                "color": colors[4],
                "linestyle": "--",
            },
        },
    },
    "files": [
        {
            "filename": "input_filter_dgdrive.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {0: "freq", 1: "lc_filter", 2: "cap_mult", 3: "z_lc", 4: "z_cap_mult"},
                "scaling": {
                    "lc_filter": lambda x: 10 ** (x["lc_filter"] / 20),
                    "cap_mult": lambda x: 10 ** (x["cap_mult"] / 20),
                    "z_lc": lambda x: 10 ** (x["z_lc"] / 20),
                    "z_cap_mult": lambda x: 10 ** (x["z_cap_mult"] / 20),
                },
            },
        },
    ],
}
