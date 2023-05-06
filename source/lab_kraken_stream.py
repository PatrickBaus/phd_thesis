import asyncio

async def process_device(device: BrickletTemperatureV2) -> None:
    """Prints the callbacks (filtered by id) of the bricklet."""
    async for temperature in device.read_temperature():
        print(f"Temperature: {temperature}")

async def shutdown(tasks: dict[asyncio.Task]) -> None:
    """Clean up by stopping all consumers"""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks)

async def main() -> None:
    """Enumerate the connection, then create workers for each device known."""
    tasks = dict()
    try:
        async with IPConnectionAsync(HOST, PORT) as connection:
            await connection.enumerate()
            async for enumeration_type, device in connection.read_enumeration():
                if device.uid == OUR_DEVICE:
                    old_task: asyncio.Task = tasks.pop(device.uid, None)
                    if old_task:
                        old_task.cancel()
                        await old_task
                    tasks[device.uid] = asyncio.create_task(process_device(device))
            # We now have a device to query for data
            async for temperature in device.read_temperature():
                print(f"Temperature: {temperature}")
    finally:
        await shutdown(tasks)


asyncio.run(main())
