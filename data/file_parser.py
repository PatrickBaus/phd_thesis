#!/usr/bin/env python
from __future__ import division

import datetime
import re
from itertools import islice

import dateutil
import numpy as np
import pandas as pd


def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


def calculate_plc100(data):
    #    groups_of_ten = pd.Series(np.repeat(range(int(len(data)/10)), 10))
    #    grouped = data.groupby(pd.Grouper(key='date', freq='400ms'))
    #    print(data.groupby(data.index // 10).mean())
    #    print(data.date.groupby(data.index // 10).first())
    new_data = data.groupby(data.index // 10).mean()
    new_data["date"] = data.date.groupby(data.index // 10).first()
    return new_data


def parse_Keysight34470A_file(filename, options, **kwargs):
    # Parse the sampling rate and start date from the header
    header = pd.read_csv(
        filename, nrows=2,
        delimiter=options.get("delimiter", ","),
        header=None
    )
    sample_interval = float(header.at[1, 1])
    # Use UTC -1 for all dates *before* 2017-05-09
    start_date = dateutil.parser.parse(
        "{date} {time} UTC".format(date=header.at[0, 1], time=header.at[0, 3])
    )
    start_date_ts = start_date.replace(tzinfo=datetime.timezone.utc).timestamp()

    # The date parser function (Timezone will be parsed as UTC)
    dateparser = lambda dates: [
        datetime.datetime.utcfromtimestamp((int(d) * sample_interval + start_date_ts))
        for d in dates
    ]

    data = pd.read_csv(
        filename,
        skiprows=3,
        header=None,
        delimiter=options.get("delimiter", ","),
        usecols=(0, 1,),
        names=("date", "value",),
        parse_dates=["date"],
        date_parser=dateparser,
    )

    data = data[abs(data.value) < 9.90000000e37]  # Drop out out bounds
    data["date"] = data["date"].dt.tz_localize("utc")

    for key, scaling_function in options.get("scaling", {}).items():
        data[key] = scaling_function(data)

    return data, {"sample_interval": sample_interval}


def parse_3458A_file(filename, options=None):
    data = pd.read_csv(
        filename,
        skiprows=1,
        header=None,
        usecols=[
            0,
        ],
        names=["voltage"],
    )

    neg = data[data["voltage"] < 0].reset_index(drop=True)
    pos = data[data["voltage"] >= 0].reset_index(drop=True)
    data = pos.join(neg, lsuffix="_ninv", rsuffix="_inv")
    data["time"] = data.index * 2 * options["sample_interval"]
    # data.set_index('time', inplace=True)
    return data


def parse_tera_term_file(filename, options=None):
    data = pd.read_csv(
        filename,
        skiprows=0,
        header=None,
        usecols=[
            1,
        ],
        names=["voltage"],
    )

    data["date"] = data.index * 2 * options["sample_interval"]
    # data.set_index('time', inplace=True)
    return data


def parse_csv_thomasS(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[
            0,
            1,
        ],
        names=["date", "voltage"],
    )
    # The date parser function (Timezone will be parsed as UTC to UTC) used for testing.
    # dateparser = lambda dates:pd.to_datetime(dates, utc=True)
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    return data


def parse_csv_thomasS_2(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["code", "date", "voltage", "inverted"],
    )  # , nrows=40000)
    # The date parser function (Timezone will be parsed as UTC to UTC) used for testing.
    # dateparser = lambda dates:pd.to_datetime(dates, utc=True)
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    return data


def parse_smi_file(filename, options, **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=0,
        usecols=[0, 1, 2, 3],
        names=["sensor_id", "date", options.get("label","sensor_value"), "unit"],
    )

    # Drop all sensor ids we do not want
    if options.get("sensor_id") is not None:
        data = data.loc[
            data["sensor_id"] == options["sensor_id"]
        ]  # Select only the sensors we want
        data.drop(columns="sensor_id", inplace=True)

    # The date parser function (Timezone will be parsed as UTC to UTC) used for testing.
    # dateparser = lambda dates:pd.to_datetime(dates, utc=True)
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    for key, scaling_function in options.get("scaling", {}).items():
        data[key] = scaling_function(data)

    return data, 0


def parse_fluke1524_file(filename, options, **kwargs):
    data = pd.read_csv(
        filename,
        delimiter=",",
        header=None,
        usecols=[1, 2, 3, 4, 5],
        names=["sensor_id", "temperature", "unit", "time", "date"],
        parse_dates={"datetime": ["date", "time"]},
    )

    # Convert to SI units
    data["temperature"] = np.where(
        (data["unit"] == "C"), data["temperature"] + 273.15, (data["temperature"] + 459.67) * 5 / 9
    )
    data.drop(columns=["unit"], inplace=True)

    # Drop all sensor ids we do not want
    if options.get("sensor_id") is not None:
        data = data.loc[
            data["sensor_id"] == options["sensor_id"]
        ]  # Select only the sensors we want
        data.drop(columns="sensor_id", inplace=True)

    # Rename datetime field
    data.rename(columns={"datetime": "date"}, inplace=True)

    data = data.reindex(columns=["date", "temperature"])
    # Data before 2018-03-15 18:00 was recorded with the wrong timezone
    if data["date"].max() > datetime.datetime(2018, 3, 15, 18):
        data["date"] = data["date"].dt.tz_localize("utc")
    else:
        data["date"] = data["date"].dt.tz_localize("Europe/Berlin").dt.tz_convert("utc")

    for key, scaling_function in options.get("scaling", {}).items():
        data[key] = scaling_function(data)

    return data, 0


def parse_RSA306_file(filename, options):
    # String looks like this:
    # >>> print(repr(line))
    # 'Resolution Bandwidth,4,Hz\n'
    regex_rbw = re.compile(
        "^Resolution Bandwidth\,([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\,([A-Za-z]+)\n$"
    )
    # String looks like this:
    # >>> print(repr(line))
    # 'Trace 1\n'
    regex_number_of_traces = re.compile("^Trace ([0-9]+)\n$")
    # String looks like this:
    # >>> print(repr(line))
    # 'XStart,0,Hz\n'
    regex_trace = re.compile(
        "^Trace ([0-9]),,([A-Za-z]+),[-+]?[0-9]*\.?[0-9]+,[-+]?[0-9]*\.?[0-9]+\n$"
    )
    # String looks like this:
    # >>> print(repr(line))
    # 'NumberPoints,64001\n'
    regex_number_of_points = re.compile("^NumberPoints\,([0-9]+)\n$")
    # String looks like this:
    # >>> print(repr(line))
    # 'XStart,0,Hz\n'
    regex_start = re.compile(
        "^XStart,([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),([A-Za-z]+)\n$"
    )
    # String looks like this:
    # >>> print(repr(line))
    # 'XStop,1000000,Hz\n'
    regex_end = re.compile(
        "^XStop,([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),([A-Za-z]+)\n$"
    )
    # String looks like this:
    # >>> print(repr(line))
    # '-30.208715438842773\n'
    # OR in the latest version
    # '0.0025581971276551485,0.000\n'
    regex_value = re.compile(
        "^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?:\,[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?\n$"
    )
    # String looks like this:
    # >>> print(repr(line))
    # '-30.208715438842773\n'
    # OR in the latest version
    # '0.0025581971276551485,0.000\n'
    regex_value = re.compile(
        "^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(\,[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?\n$"
    )

    selected_trace = options.get("trace", 1)

    rbw = None
    number_of_traces = 0
    start_of_data = None

    number_of_points = None
    start_point = None
    end_point = None
    has_index = None
    trace_unit = None
    with open(filename) as lines:
        enumerated_lines = enumerate(lines)  # We need the itarator to skip lines
        for row_num, line in enumerated_lines:
            if rbw is None and regex_rbw.match(line):
                rbw = float(regex_rbw.match(line).group(1))
                unit = regex_rbw.match(line).group(2)
                print(f"  Resolution Bandwidth: {rbw} {unit}")
                continue
            if regex_number_of_traces.match(line):
                # We need to enter this regex multiple times, because there is no distinct line that has the number
                # of rows, so we misuse the trace parameters to find the highest trace number
                number_of_traces = max(
                    int(regex_number_of_traces.match(line).group(1)), number_of_traces
                )
                continue
            if start_of_data is None and line == "[Traces]\n":
                print(
                    f"  Number of traces in file: {number_of_traces}"
                )  # We now know the number of total traces
                if selected_trace > number_of_traces:
                    raise TypeError(
                        "Selected trace is larger than the number of available traces!"
                    )
                next(enumerated_lines)  # Skip '[Trace]\n'
            if regex_trace.match(line):
                # We have found a trace
                # parse the header and check whether we need it
                match = regex_trace.match(line)
                current_trace, trace_unit = int(match.group(1)), match.group(2)
                row_num, line = next(enumerated_lines)
                number_of_points = int(regex_number_of_points.match(line).group(1))
                if current_trace != selected_trace:
                    # skip number_of_points +2 two rows of the header
                    islice(enumerated_lines, number_of_points + 2, None)
                else:
                    row_num, line = next(enumerated_lines)
                    match = regex_start.match(line)
                    start_point = float(match.group(1))
                    unit = match.group(2)
                    print(f"  Starting point: {start_point} {unit}")
                    row_num, line = next(enumerated_lines)
                    end_point = float(regex_end.match(line).group(1))
                    print(f"  End point: {end_point} {unit}")
                    # Next row is the first trace
                    row_num, line = next(enumerated_lines)
                    has_index = regex_value.match(line).group(2)
                    start_of_data = row_num
                    break

    #    with open(filename) as lines:
    #        for row_num, line in enumerate(islice(lines, start_of_data, None)):
    #            if start_point is None and regex_start.match(line):
    #                start_point = float(regex_start.match(line).group(1))
    #                unit = regex_start.match(line).group(2)
    #                print(f"  Starting point: {start_point} {unit}")
    #                continue
    #            if number_of_points is None and regex_number_of_points.match(line):
    #                number_of_points = int(regex_number_of_points.match(line).group(1))
    #                continue
    #            if end_point is None and regex_end.match(line):
    #                end_point = float(regex_end.match(line).group(1))
    #                print(f"  End point: {end_point} {unit}")
    #                continue
    #            if regex_value.match(line):
    #                has_index = regex_value.match(line).group(2)
    #                start_of_data = row_num + start_of_data
    #                break

    data = None
    if has_index is not None:
        data = pd.read_csv(
            filename,
            delimiter=",",
            header=None,
            skiprows=start_of_data,
            nrows=number_of_points,
            usecols=[0, 1],
            names=["psd", "frequency"],
            index_col=1,
        )
    else:
        data = pd.read_csv(
            filename,
            delimiter=",",
            header=None,
            skiprows=start_of_data,
            nrows=number_of_points,
            usecols=[
                0,
            ],
            names=[
                "psd",
            ],
        )
        step_size = (end_point - start_point) / (len(data.index) - 1)
        data.index *= step_size
        data.index.name = "frequency"

    if trace_unit == "dBm":
        data.psd = 10 ** ((data.psd - 10) / 20) / np.sqrt(2)

    data.psd /= np.sqrt(rbw) * options.get("gain", 1)
    return data, {"type": "spectrum"}


def parse_MSO9000_file(filename, options):
    data = pd.read_csv(filename, delimiter=",", usecols=[0, 1], names=["date", "value"])
    data.value /= options.get("gain", 1)

    data = data.set_index("date")
    sample_interval = (data.index[-1] - data.index[0]) / (len(data.index) - 1)
    print(
        "  Infiniium sampling rate: {sampling_rate} Hz".format(
            sampling_rate=1 / sample_interval
        )
    )
    return data, {"sample_interval": sample_interval}


def parse_3458A_SN18_file_1(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=[
            "date",
            "value",
            "temp",
        ],
    )
    sample_interval = (data.index[-1] - data.index[0]) / (len(data.index) - 1)

    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    return data, 0


def parse_3458A_SN18_file_2(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 3],
        names=[
            "date",
            "value",
            "temp",
        ],
    )
    sample_interval = (data.index[-1] - data.index[0]) / (len(data.index) - 1)

    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    return data, 0


def parse_3458A_SN18_file_3(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 4],
        names=[
            "date",
            "value2",
            "value",
            "temp",
        ],
    )
    sample_interval = (data.index[-1] - data.index[0]) / (len(data.index) - 1)

    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    return data, 0


def parse_3458A_SN18_file_4(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["date", "cal71", "cal72", "temp", "tempPt100", "temp10k"],
    )
    sample_interval = (data.index[-1] - data.index[0]) / (len(data.index) - 1)

    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    if options.get("sensor_id") is not None:
        data.rename(columns={options["sensor_id"]: "value"}, inplace=True)

    return data, 0


def parse_3458A_SN18_file_5(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 3],
        names=["date", "value", "temp10k"],
    )
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    gain = options.get("gain", 1)
    if gain != 1:
        data.value /= gain

    return data, 0  # TODO Add sample interval


def parse_mecom_file(filename, options=None):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 3],
        names=["date", "tec_sensor", "tec_current"],
    )

    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    #  data.set_index('date', inplace=True)

    return data, 0


def parse_3458A_dgDrive_file(filename, options):
    data = pd.read_csv(
        filename, comment="#", header=None, usecols=[0, 1], names=["date", "value"]
    )
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    gain = options.get("gain", 1)
    if gain != 1:
        data.value /= gain

    # data.set_index('date', inplace=True)
    #    sampling_interval = data['date'].diff().mean().total_seconds()
    return data, 0  # TODO Add sample interval


def parse_3458A_5440B_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["date", "3458A", "dmm_temp", "DMM6500", "34470A"],
    )
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    #  if options.get('sensor_id') is not None:
    #    data = data[['date',options['sensor_id']]]
    #    data.rename(columns={options['sensor_id']: 'value'}, inplace=True)

    return data, 0  # TODO Add sample interval


def parse_3458A_5440B_v2_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["date", "dmm_temp", "DMM6500", "34470A"],
    )
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    #  if options.get('sensor_id') is not None:
    #    data = data[['date',options['sensor_id']]]
    #    data.rename(columns={options['sensor_id']: 'value'}, inplace=True)

    return data, 0  # TODO Add sample interval


def parse_3458A_tempco_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["date", "value", "ambient", "dmm"],
    )
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    # data.set_index('date', inplace=True)
    #    sampling_interval = data['date'].diff().mean().total_seconds()

    gain_settings = options.get("gain", {})
    for key in gain_settings:
        if key in data and gain_settings[key] != 1:
            data[key] /= gain_settings[key]

    return data, 0


def parse_3458A_tempco_v2_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["date", "value", "dmm", "ambient", "humidity"],
    )
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    if options.get("sensor_id") is not None:
        data = data[["date", options["sensor_id"]]]
        data.rename(columns={options["sensor_id"]: "value"}, inplace=True)

    gain = options.get("gain", 1)
    if gain != 1:
        data.value /= gain

    return data, 0


def convertResistanceToTemperature(values):
    # Constants for Amphenol DC95 (Material Type 10kY)
    a = 3.3540153 * 10**-3
    b = 2.7867185 * 10**-4
    c = 4.0006637 * 10**-6
    d = 1.5575628 * 10**-7
    rt25 = 10 * 10**3

    return (
        1
        / (
            a
            + b * np.log(values / rt25)
            + c * np.log(values / rt25) ** 2
            + d * np.log(values / rt25) ** 3
        )
        - 273.15
    )


def parse_3458A_tempco_v3_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["date", "value", "dmm", "ambient", "humidity", "shunt"],
    )
    data.shunt = convertResistanceToTemperature(data.shunt)
    # data = data[data.value != -1.000000000E+38]
    data["date"] = pd.to_datetime(
        data["date"], utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    if options.get("sensor_id") is not None:
        data = data[["date", options["sensor_id"]]]
        data.rename(columns={options["sensor_id"]: "value"}, inplace=True)

    gain = options.get("gain", 1)
    if gain != 1:
        data.value /= gain

    return data, 0


def parse_3458A_tempco_v4_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6],
        names=["date", "value", "dmm", "ambient", "humidity", "ambient2", "shunt"],
    )
    #  data.shunt = convertResistanceToTemperature(data.shunt)
    # data = data[data.value != -1.000000000E+38]
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    #  data.set_index('date', inplace=True)

    if options.get("sensor_id") is not None:
        data = data[["date", options["sensor_id"]]]
        data.rename(columns={options["sensor_id"]: "value"}, inplace=True)

    offset = options.get("offset", 0)
    if offset != 0:
        data.value += offset

    gain = options.get("gain", 1)
    if gain != 1:
        data.value /= gain

    return data, 0


def parse_3458A_tempco_v5_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        names=[
            "date",
            "value",
            "HP3458A",
            "dmm",
            "ambient",
            "humidity",
            "ambient2",
            "DMM6500",
        ],
    )
    data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    #  data = data[data.value != -1.000000000E+38]
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    #  data.set_index('date', inplace=True)

    #  offset =  options.get('offset', 0)
    #  if offset != 0:
    #    data.value += offset

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    #  gain_settings =  options.get('gain', {})
    #  for key in gain_settings:
    #    if key in data and gain_settings[key] != 1:
    #      data[key] /= gain_settings[key]

    #  data.value = 1 / data.value * 7.5
    #  data.HP3458A = 1 / data.HP3458A * 7.5

    return data, 0


def parse_3458A_tempco_v6_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["date", "HP3458A", "dmm", "ambient", "humidity", "ambient2"],
    )
    #  data = data[data.value != -1.000000000E+38]
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3458A_tempco_v7_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6],
        names=["date", "HP3458A", "ambient", "dmm", "humidity", "ambient2", "DMM6500"],
        na_values="None",
    )
    #  data = data[data.value != -1.000000000E+38]
    data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3458A_tempco_v8_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["date", "HP3458A", "temp_dmm", "temp_dut", "DMM6500"],
        na_values="None",
    )
    #  data = data[data.value != -1.000000000E+38]
    # data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3458A_tempco_v9_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6],
        names=[
            "date",
            "HP3458A",
            "temp_dmm",
            "temp_dut",
            "humidity",
            "temp_ambient",
            "DMM6500",
        ],
        na_values="None",
    )
    #  data = data[data.value != -1.000000000E+38]
    # data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3458A_tempco_v11_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        names=[
            "date",
            "HP3458A",
            "temp_dmm",
            "temp_dut",
            "humidity",
            "temp_ambient",
            "DMM6500",
            "tec_sensor",
            "tec_current",
            "tec_voltage",
            "setpoint",
        ],
        na_values="None",
    )
    #  data = data[data.value != -1.000000000E+38]
    # data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3458A_tempco_v12_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        names=[
            "date",
            "HP3458A",
            "temp_dmm",
            "temp_dut",
            "humidity_dut",
            "temp_ambient",
            "DMM6500",
            "K2002",
            "tec_sensor",
            "tec_current",
            "tec_voltage",
            "setpoint",
        ],
        na_values="None",
    )
    #  data = data[data.value != -1.000000000E+38]
    # data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    data = data[2:]  # The first 2 values of the HP3458A are a few ppm out
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3458A_tempco_v13_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        names=[
            "date",
            "HP3458A",
            "temp_dmm",
            "temp_dut",
            "humidity_ambient",
            "humidity_dut",
            "temp_ambient",
            "DMM6500",
            "K2002",
            "tec_sensor",
            "tec_current",
            "tec_voltage",
            "setpoint",
        ],
        na_values="None",
    )
    #  data = data[data.value != -1.000000000E+38]
    # data.DMM6500 = convertResistanceToTemperature(data.DMM6500)
    data = data[2:]  # The first 2 values of the HP3458A are a few ppm out
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_LM399_logger_file(filename, options):
    data = pd.read_csv(
        filename, comment="#", header=None, usecols=[0, 1], names=["date", "value"]
    )
    data = data[data.value > -9.90000000e37]  # Drop out out bounds
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    remove_beyond_sigma = options.get("remove_outliers", {}).get("sigma")
    if remove_beyond_sigma is not None:
        sigma = data.value.std()
        print(
            "    Removing outliers beyond {sigma_count}σ (σ = {sigma})".format(
                sigma_count=remove_beyond_sigma, sigma=sigma
            )
        )
        data = data[
            np.abs(data.value - data.value.mean()) <= (remove_beyond_sigma * sigma)
        ]
        print(
            "    Std. deviation after removing outliers σ = {sigma}".format(
                sigma=data.value.std()
            )
        )

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_LM399_logger_v2_file(filename, options, **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["date", "value", "tmp236"],
    )
    data = data[abs(data.value) < 9.90000000e37]  # Drop out out bounds
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    #  data = calculate_plc100(data)

    remove_beyond_sigma = options.get("remove_outliers", {}).get("sigma")
    if remove_beyond_sigma is not None:
        sigma = data.value.std()
        print(
            "    Removing outliers beyond {sigma_count}σ (σ = {sigma})".format(
                sigma_count=remove_beyond_sigma, sigma=sigma
            )
        )
        data = data[
            np.abs(data.value - data.value.mean()) <= (remove_beyond_sigma * sigma)
        ]
        print(
            "    Std. deviation after removing outliers σ = {sigma}".format(
                sigma=data.value.std()
            )
        )

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_LTZ1000_logger_file(filename, options):
    data = pd.read_csv(
        filename, comment="#", header=None, usecols=[0, 1], names=["date", "HP3458A"]
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_LTZ1000_logger_v2_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["date", "HP3458A", "ambient", "dmm", "humidity"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_LTZ1000_logger_v3_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        names=[
            "date",
            "HP3458A",
            "temp_10k",
            "temp_100",
            "humidity_lab",
            "humidity_dut",
            "temp_ee07",
            "K2002",
            "tec_sensor",
            "tec_current",
            "tec_voltage",
            "setpoint",
        ],
        dtype={"HP3458A": "float"},
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_3478A_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["date", "HP3478A", "temp_dut"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    if options.get("convert_temperature", False):
        data.HP3478A = convertResistanceToTemperature(data.HP3478A)

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_RTH1004_file(filename, options):
    data = pd.read_csv(
        filename, delimiter=",", usecols=[0, 1], skiprows=22, names=["date", "value"]
    )
    data.value /= options.get("gain", 1)

    data = data.set_index("date")
    sample_interval = (data.index[-1] - data.index[0]) / (len(data.index) - 1)
    print(
        "  RTH1004 sampling rate: {sampling_rate} Hz".format(
            sampling_rate=1 / sample_interval
        )
    )
    return data, {"sample_interval": sample_interval}


def parse_RTH1004_spectrum_file(filename, options):
    # String looks like this:
    # >>> print(repr(line))
    # 'Resolution Bandwidth,4,Hz\n'
    regex_rbw = re.compile("^RBW \[([A-Za-z]+)\],([0-9]*)\n$")

    rbw = None
    with open(filename) as lines:
        for row_num, line in enumerate(lines):

            if rbw is None and regex_rbw.match(line):
                rbw = float(regex_rbw.match(line).group(2))
                unit = regex_rbw.match(line).group(1)
                print("  Resolution Bandwidth: {rbw} {unit}".format(rbw=rbw, unit=unit))

    data = pd.read_csv(
        filename,
        delimiter=",",
        usecols=[0, 4],
        skiprows=18,
        names=["frequency", "psd"],
        index_col=0,
    )
    data.psd /= np.sqrt(rbw) * options.get("gain", 1)
    return data, {"type": "spectrum"}


def parse_Fluke5440B_test_file(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6],
        names=[
            "date",
            "HP3458A",
            "temp_dmm",
            "temp_dut",
            "humidity",
            "temp_ambient",
            "K2002",
        ],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_slice_qtc_file(filename, options):
    data = pd.read_csv(
        filename, comment="#", header=None, usecols=[0, 1], names=["date", "slice_qtc"]
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_Fluke5440B_test_file_v2(filename, options):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["date", "HP3458A", "temp_10k", "temp_100", "humidity_lab", "K2002"],
        dtype={"HP3458A": "float"},
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_Keysight34470A_file_2(filename, options, delimiter=","):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["date", "34470A", "temp_100", "humidity"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data[1:], 0


def parse_labtemp_drift_file(filename, options, delimiter=","):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["date", "temp_10k", "temp_100", "humidity_lab", "temp_ee07"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_rth_digital_file(filename, options, delimiter=","):
    data = pd.read_csv(
        filename,
        skiprows=22,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        names=["date", "D7", "D6", "D5", "D4", "D3", "D2", "D1", "D0"],
    )
    # data.date = pd.to_datetime(data.date, utc=True)   # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_SCAN2000_file(filename, options, delimiter=",", **_kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=range(19),
        names=[
            "date",
        ]
        + [f"K2002 CH{channel+1}" for channel in range(10)]
        + [
            "temp_10k",
            "temp_100",
            "humidity",
            "temp_chamber",
            "temp_tec",
            "current_tec",
            "voltage_tec",
            "setpoint_tec",
        ],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_WS8_file(filename, options, delimiter=","):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=[
            0,
            3,
            4,
            5,
            6,
        ],
        names=["date", "ch1", "ch2", "pressure", "temp"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file
    # Convert to Hz
    data.ch1 *= 10**12
    data.ch2 *= 10**12
    # data.ch3 *= 10**12
    data.pressure *= 100  # Convert to Pa

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data[key])

    return data, 0


def parse_timescale_db_file(filename, options, delimiter=","):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=range(7),
        names=[
            "date",
            "humidity_dut",
            "temp_table",
            "laser_power",
            "air_pressure",
            "frequency",
            "piezo_voltage",
        ],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_timescale_db_015_file(filename, options, delimiter=","):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=range(6),
        names=[
            "date",
            "humidity_dut",
            "temp_table",
            "air_pressure",
            "diode_voltage",
            "piezo_voltage",
        ],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_timescale_db_file_v2(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=range(3),
        names=[
            "date",
            "output",
            "temperature",
        ],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_timescale_db_file_v3(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=range(4),
        names=["date", "output", "temperature_room", "temperature_labnode"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_timescale_db_file_v4(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=range(len(options["labels"])+1),
        names=["date",] + options["labels"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_timescale_db_fluke5440b(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        skiprows=1,
        comment="#",
        header=None,
        usecols=range(6),
        names=["date", "34470a", "3458a", "k2002", "dmm6500", "temperature"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_data_logger_fluke5440b(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 4, 6],
        names=["date", "k2002", "3458a", "34470a", "dmm6500", "temperature"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_data_logger_short_dmm(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1,],
        names=["date", options.get("label", "dmm")],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0


def parse_data_logger_short_dmm_frank(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1,],
        names=["date", options.get("label", "dmm")],
    )
    data.date = pd.to_timedelta(data.date, unit='s')
    data.date = pd.to_datetime(
        "1970-01-01", utc=True,
    )  + pd.to_timedelta(data.date, unit='s')  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        if key in data:
            data[key] = scaling_function(data)

    return data, 0

def noise_gen(filename, options, **kwargs):
    from random import seed

    seed(42)

    amplitude = options.get("amplitude", 1)
    noise_beta = {
        "blue": 1,
        "white": 0,
        "pink": -1,
        "brown": -2,
        "running": -3,
    }
    beta = noise_beta[options.get("noise_type", "white")]
    N = options.get("samples", 1e5)
    x = np.arange(0, N / 2 + 1)
    mag = amplitude * x ** (beta / 2) * np.random.randn(int(N / 2 + 1))
    pha = 2 * np.pi * np.random.rand(int(N / 2 + 1))
    real = mag * np.cos(pha)
    imag = mag * np.sin(pha)
    real[0] = 0
    imag[0] = 0

    c = real + 1j * imag
    tod = np.fft.irfft(c)

    df = pd.DataFrame({"date": np.arange(N), "value": tod})
    return df, 0

def parse_data_dgdrive_powermeter(filename, options, delimiter=",", **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        usecols=[0, 1,2,3],
        names=["date", "dgdrive", "pm400", "34470a"],
    )
    data.date = pd.to_datetime(
        data.date, utc=True
    )  # It is faster to parse the dates *after* parsing the csv file

    for key, scaling_function in options.get("scaling", {}).items():
        data[key] = scaling_function(data)


    return data, 0

def parse_data_ltspice_fets(filename, options, **kwargs):
    data = pd.read_csv(
        filename,
        comment="#",
        header=None,
        skiprows=options.get("skiprows", 1),
        delimiter=options.get("delimiter", ","),
        usecols=options["columns"].keys(),
        names=options["columns"].values(),
    )

    for key, scaling_function in options.get("scaling", {}).items():
        data[key] = scaling_function(data)

    return data, 0

def parse_bode100_file(filename, options, **kwargs):
    starting_row = None
    final_row = None
    selected_trace = options.get("trace", 0)
    current_trace = 0

    with open(filename) as lines:
        for row_num, line in enumerate(lines):
            if starting_row is None and line == "------\n":
                starting_row = row_num + 2  # Skip this row + header
            if starting_row is not None and line == "\n":
                if current_trace == selected_trace:
                    final_row = row_num
                    break
                else:
                    starting_row = row_num + 2  # Skip this row + header
                    current_trace += 1

    if starting_row is None or final_row is None or selected_trace != current_trace:
        raise Exception(f"Invalid data file or incorrect trace selected: {filename}")

    data = pd.read_csv(
        filename,
        skiprows=starting_row,
        nrows=final_row-starting_row,
        delimiter=",",
        usecols=options["columns"].keys(),
        names=options["columns"].values(),
    )

    for key, scaling_function in options.get("scaling", {}).items():
        data[key] = scaling_function(data)

    return data,0


FILE_PARSER = {
    "34470A": parse_Keysight34470A_file,
    "34470A_resistance": parse_Keysight34470A_file_2,
    "3458A_DgTemp": parse_3458A_file,
    "tera-term": parse_tera_term_file,
    "csv-thomasS": parse_csv_thomasS,
    "csv-thomasS_2.0": parse_csv_thomasS_2,
    "fluke1524": parse_fluke1524_file,
    "smi": parse_smi_file,
    "MSO9000": parse_MSO9000_file,
    "RSA306": parse_RSA306_file,
    "3458A_SN18_1.0": parse_3458A_SN18_file_1,
    "3458A_SN18_2.0": parse_3458A_SN18_file_2,
    "3458A_SN18_3.0": parse_3458A_SN18_file_3,
    "3458A_SN18_4.0": parse_3458A_SN18_file_4,
    "3458A_SN18_5.0": parse_3458A_SN18_file_5,
    "3458A_DgDrive": parse_3458A_dgDrive_file,
    "3458A_Tempco": parse_3458A_tempco_file,
    "3458A_Tempco_v2": parse_3458A_tempco_v2_file,
    "3458A_Tempco_v3": parse_3458A_tempco_v3_file,
    "3458A_Tempco_v4": parse_3458A_tempco_v4_file,
    "3458A_Tempco_v5": parse_3458A_tempco_v5_file,
    "3458A_Tempco_v6": parse_3458A_tempco_v6_file,
    "3458A_Tempco_v7": parse_3458A_tempco_v7_file,
    "3458A_Tempco_v8": parse_3458A_tempco_v8_file,
    "3458A_Tempco_v9": parse_3458A_tempco_v9_file,
    "3458A_Tempco_v11": parse_3458A_tempco_v11_file,
    "3458A_Tempco_v12": parse_3458A_tempco_v12_file,
    "3458A_Tempco_v13": parse_3458A_tempco_v13_file,
    "mecom": parse_mecom_file,
    "3458A_Fluke5440B": parse_3458A_5440B_file,
    "3458A_Fluke5440B_v2": parse_3458A_5440B_v2_file,
    "LM399_logger": parse_LM399_logger_file,
    "LM399_logger_v2": parse_LM399_logger_v2_file,
    "LTZ1000_logger": parse_LTZ1000_logger_file,
    "LTZ1000_logger_v2": parse_LTZ1000_logger_v2_file,
    "LTZ1000_logger_v3": parse_LTZ1000_logger_v3_file,
    "3478A": parse_3478A_file,
    "RTH1004": parse_RTH1004_file,
    "RTH1004_spectrum": parse_RTH1004_spectrum_file,
    "Fluke5440B_test": parse_Fluke5440B_test_file,
    "slice_qtc": parse_slice_qtc_file,
    "Fluke5440B_test_v2": parse_Fluke5440B_test_file_v2,
    "Kraken": parse_labtemp_drift_file,
    "rth_digital": parse_rth_digital_file,
    "scan2000": parse_SCAN2000_file,
    "ws8": parse_WS8_file,
    "timescale_db": parse_timescale_db_file,
    "timescale_db_015": parse_timescale_db_015_file,
    "timescale_db_2": parse_timescale_db_file_v2,
    "timescale_db_3": parse_timescale_db_file_v3,
    "timescale_db_4": parse_timescale_db_file_v4,
    "timescale_db_fluke5440b": parse_timescale_db_fluke5440b,
    "data_logger_fluke5440b": parse_data_logger_fluke5440b,
    "data_logger_short_dmm": parse_data_logger_short_dmm,
    "data_logger_short_dmm_frank": parse_data_logger_short_dmm_frank,
    "noise_gen": noise_gen,
    "dgdrive_powermeter": parse_data_dgdrive_powermeter,
    "ltspice_fets": parse_data_ltspice_fets,
    "bode100": parse_bode100_file,
}


def parse_file(parser, filename, **kwargs):
    return FILE_PARSER[parser](filename=filename, **kwargs)
