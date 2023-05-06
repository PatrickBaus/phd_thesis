ipcon = IPConnection()

# Callback function for the temperature callback
def cb_temperature(temperature: int) -> None:
    """Read the temperature data from the device and print it."""
    print(f"Temperature: {temperature/100.0} °C")

def cb_connected(connect_reason: int) -> None:
    """Query for sensors as soon as the connection is established."""
    ipcon.enumerate()

def cb_enumerate(uid: str, *_args) -> None:
    if uid == OUR_KNOWN_DEVICE:
        dev = BrickletTemperatureV2(uid, ipcon)
        # Register temperature callback to function cb_temperature
        dev.register_callback(dev.CALLBACK_TEMPERATURE, cb_temperature)
        dev.set_temperature_callback_configuration(1000, False, "x", 0, 0)

ipcon.connect(HOST, PORT)  # blocking call
# Register enumerate callback
ipcon.register_callback(IPConnection.CALLBACK_ENUMERATE, cb_enumerate)
ipcon.register_callback(IPConnection.CALLBACK_CONNECTED, cb_connected)

input("Press key to exit\n")
ipcon.disconnect()
