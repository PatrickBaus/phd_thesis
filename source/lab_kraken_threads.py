ipcon = IPConnection()


def cb_temperature(temperature: int) -> None:
    """Read the temperature data from the device and print it."""
    print(f"Temperature: {temperature/100.0} °C")


def cb_connected(connect_reason: int) -> None:
    """Query for sensors as soon as the connection is established."""
    ipcon.enumerate()


def cb_enumerate(uid: str, *_args) -> None:
    """Search for OUR_KNOWN_DEVICE and, when found, read from it."""
    if uid == OUR_KNOWN_DEVICE:
        device = BrickletTemperatureV2(uid, ipcon)
        # Register temperature callback as function cb_temperature
        device.register_callback(device.CALLBACK_TEMPERATURE, cb_temperature)
        device.set_temperature_callback_configuration(1000, False, "x", 0, 0)


# Connect to the sensor host
ipcon.connect(HOST, PORT)
# Register connection and enumeration callbacks
ipcon.register_callback(IPConnection.CALLBACK_ENUMERATE, cb_enumerate)
ipcon.register_callback(IPConnection.CALLBACK_CONNECTED, cb_connected)

input("Press key to exit\n")
ipcon.disconnect()
