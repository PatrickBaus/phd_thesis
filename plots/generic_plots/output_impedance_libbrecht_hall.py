import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "Output impedance Libbrecht & Hall current source",
    "show": True,
    "output_file": {"fname": "../images/output_impedance_libbrecht_hall.pgf"},
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "crop_secondary_to_primary": True,
    "legend_position": "best",
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Frequency in \unit{\Hz}",
            "y_label": r"Output impedance in \unit{\ohm}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": 6,
            "x_scale": "log",
            "y_scale": "log",
        },
        "x-axis": "frequency",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "noC": {
                "label": r"no $C_1$",
                "color": colors[0],
            },
            "1u": {
                "label": r"$C_1 = \qty{1}{\uF}$",
                "color": colors[1],
            },
        },
    },
    "files": [
        {
            "filename": "modulation_input_LibrechtHall.txt",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {0: "frequency", 1: "noC"},
                "scaling": {
                    "noC": lambda x: 10 ** (x["noC"] / 20),
                },
            },
        },
        {
            "filename": "modulation_input_LibrechtHall_1u.txt",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {0: "frequency", 1: "1u"},
                "scaling": {
                    "1u": lambda x: 10 ** (x["1u"] / 20),
                },
            },
        },
    ],
}
