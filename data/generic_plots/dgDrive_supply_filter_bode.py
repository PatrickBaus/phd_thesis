import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "Supply Filter Transfer function",
    "show": True,
    "output_file": {"fname": "../images/dgDrive_supply_filter_bode.pgf"},
    "legend_position": "lower left",
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Frequency in \unit{\Hz}",
            "y_label": r"Magnitude in \unit{\dB}",
            "invert_x": False,
            "invert_y": False,
            "x_scale": "log",
        },
        "x-axis": "frequency",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "magnitude": {
                "label": "LC Filter",
                "color": colors[1],
            },
            "lc_filter": {
                "label": "Simulation",
                "color": colors[0],
            },
        },
    },
    "files": [
        {
            "filename": "DgDrive PSRR_take2_2023-02-18T01_12_08.csv",
            "show": True,
            "parser": "bode100",
            "options": {
                "trace": 1,
                "columns": {
                    0: "frequency",
                    1: "magnitude",
                },
                "scaling": {
                    "magnitude": lambda x: x["magnitude"] - 40,
                },
            },
        },
        {
            "filename": "input_filter_dgdrive.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {0: "frequency", 1: "lc_filter"},
                "scaling": {
                    "lc_filter": lambda x: x["lc_filter"][(x["frequency"] >= 100) & (x["frequency"] <= 1e6)],
                },
            },
        },
    ],
}
