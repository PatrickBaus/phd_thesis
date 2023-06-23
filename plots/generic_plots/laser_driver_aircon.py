import seaborn as sns

colors = sns.color_palette("colorblind")
phi = (5**0.5 - 1) / 2  # golden ratio
plot = {
    "description": "Lab 011 Temperature Durfee (positive)",
    "show": True,
    "output_file": {"fname": "../images/laser_driver_aircon.pgf"},
    "plot_size": (441.01773 / 72.27 * 0.89, 441.01773 / 72.27 * 0.89 * phi),
    "crop": {
        "crop_index": "date",
        "crop": ["2017-04-05 00:00:00", "2017-04-06 00:00:00"],
    },
    "crop_secondary_to_primary": False,
    "primary_axis": {
        "axis_settings": {
            "y_label": r"Current deviation in \unit{\A}",
            "x_label": r"Time (UTC)",
            "invert_x": False,
            "invert_y": False,
            "fixed_order": -6,
            "x_scale": "time",
            "date_format": "%H:%M",
        },
        "x-axis": "date",
        "plot_type": "absolute",  # absolute, relative, proportional
        "columns_to_plot": {
            "value": {
                "label": "Output current",
                "color": colors[0],
            },
        },
    },
    "secondary_axis": {
        "show": True,
        "axis_settings": {
            "y_label": r"Temperature in \unit{\celsius}",
            "invert_y": False,
            "fixed_order": None,
        },
        "x-axis": "date",
        "columns_to_plot": {
            "temperature": {
                "label": "Ambient Temperature",
                "color": sns.color_palette("bright")[3],
            },
        },
    },
    "files": [
        {
            "filename": "Durfee_pos_tantalum_36h.csv",
            "show": True,
            "parser": "34470A",
            "options": {
                "scaling": {
                    "value": lambda x: x["value"] - x["value"].min(),
                },
            },
        },
        {
            "filename": "sensorData_2017-04-04 18-13-00_2017-04-06 06-13-00.csv",
            "show": True,
            "parser": "smi",
            "options": {
                "sensor_id": 8,
                "label": "temperature",
                "scaling": {
                    "temperature": lambda x: x["temperature"] - 273.15,
                },
            },
        },
    ],
}
