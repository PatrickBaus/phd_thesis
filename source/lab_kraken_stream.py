import asyncio
from aiostream import stream, pipe


async def main() -> None:
    """Define a stream, then execute it."""
    async with IPConnectionAsync(HOST, PORT) as connection:
        connection.enumerate()
        reader = (
            stream.iterate(connection.read_enumeration())  # read devices
            | pipe.filter(lambda device: device.uid == OUR_DEVICE)  # keep our device
            | pipe.switchmap(lambda device: device.read_temperature())  # read data
            | pipe.print("Temperature: {} °C")  # Print results
        )
        await reader  # start the stream


asyncio.run(main())
