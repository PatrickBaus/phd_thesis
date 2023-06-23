import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "Output impedance simulation",
    "show": True,
    "output_file": {"fname": "../images/ltspice_output_impedance_simulation.pgf"},
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Drain-source voltage $V_{DS}$ in \unit{\V}",
            "y_label": r"Ouput Impedance $R_{out}$ in \unit{\ohm}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": 9,
            "y_scale": "log",
            # "x_scale": "log",  # Turn this on to show, that R_out is a power law
        },
        "x-axis": "vds",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "rout": {
                "label": r"DC",
                "color": colors[0],
            },
            "rout10MegHz": {
                "label": r"\qty{1}{\MHz}",
                "color": colors[1],
            },
        },
    },
    "files": [
        {
            "filename": "mosfet_current_source_output_impedance.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {0: "vload", 1: "rout", 2: "rout10MegHz"},
                "scaling": {
                    "vds": lambda x: 3.5 - x["vload"],
                    "rout": lambda x: 10 ** (x["rout"] / 20),
                    "rout10MegHz": lambda x: 10 ** (x["rout10MegHz"] / 20),
                },
            },
        },
    ],
}
