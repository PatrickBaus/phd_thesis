ipcon = IPConnection()
devices = dict()

# Callback function for the temperature callback
def cb_temperature(temperature):
    print("Temperature: " + str(temperature/100.0) + " degC")

def cb_connected(connect_reason)::
    ipcon.enumerate()

def cb_disconnected(disconnect_reason)::
    log_reason(disconnect_reason)

def cb_enumerate(uid, *_args):
    if uid == OUR_KNOWN_DEVICE:
        dev = BrickletTemperatureV2(uid, ipcon)
        # Register temperature callback to function cb_temperature
        dev.register_callback(dev.CALLBACK_TEMPERATURE, cb_temperature)
        dev.set_temperature_callback_configuration(1000, False, "x", 0, 0)
        devives[uid] = dev

if __name__ == "__main__":
    ipcon.connect(HOST, PORT)  # blocking call
    # Register enumerate callback
    ipcon.register_callback(IPConnection.CALLBACK_ENUMERATE, cb_enumerate)
    ipcon.register_callback(IPConnection.CALLBACK_CALLBACK_CONNECTED, cb_connected)
    ipcon.register_callback(IPConnection.CALLBACK_CALLBACK_DISCONNECTED, cb_disconnected)

    input("Press key to exit\n") # Use raw_input() in Python 2
    ipcon.disconnect()
