import asyncio

async def process_device(device: BrickletTemperatureV2) -> None:
    """Prints the callbacks (filtered by id) of the bricklet."""
    async for temperature in device.read_temperature():
        print(f"Temperature: {temperature} °C")

async def shutdown(tasks: dict[asyncio.Task]) -> None:
    """Clean up by stopping all consumers"""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks)

async def main() -> None:
    """Enumerate the connection, then create workers for each device known."""
    try:
        async with IPConnectionAsync(HOST, PORT) as connection:
            await connection.enumerate()
            async for enumeration_type, device in connection.read_enumeration():
                if device.uid == OUR_DEVICE:
                    asyncio.create_task(process_device(device))
    finally:
        await shutdown(tasks)


asyncio.run(main())
