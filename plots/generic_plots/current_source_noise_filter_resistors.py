import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "Current Source noise (different R_filt)",
    "show": True,
    "output_file": {"fname": "../images/current_source_noise_filter_resistors.pgf"},
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "crop_secondary_to_primary": True,
    "legend_position": "best",
    "primary_axis": {
        "axis_settings": {
            "x_label": r"Frequency in \unit{\Hz}",
            "y_label": r"Noise density in \unit[power-half-as-sqrt,per-mode=symbol]{\A \Hz\tothe{-0.5}}",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -9,
            "x_scale": "log",
            "y_scale": "lin",
        },
        "x-axis": "frequency",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "249ohm": {
                "label": r"$R_{filt} = \qty{249}{\ohm}$",
                "color": colors[0],
            },
            "510ohm": {
                "label": r"$R_{filt} = \qty{510}{\ohm}$",
                "color": colors[1],
            },
            "1000ohm": {
                "label": r"$R_{filt} = \qty{1}{\kilo\ohm}$",
                "color": colors[2],
            },
            "1500ohm": {
                "label": r"$R_{filt} = \qty{1.5}{\kilo\ohm}$",
                "color": colors[3],
            },
        },
    },
    "files": [
        {
            "filename": "current_regulator_v3_AD797_noise.csv",
            "show": True,
            "parser": "ltspice_fets",
            "options": {
                "columns": {
                    0: "frequency",
                    1: "249ohm",
                    2: "510ohm",
                    3: "1000ohm",
                    4: "1500ohm",
                },
                "scaling": {},
            },
        },
    ],
}
