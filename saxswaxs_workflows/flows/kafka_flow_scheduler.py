import json
import asyncio
import logging
import os

from kafka import KafkaConsumer

from prefect.blocks.system import JSON
from utils_prefect import _schedule
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger("data_watcher")
logger.setLevel("INFO")
logname = "data_watcher_p03"
logger.addHandler(logging.StreamHandler())


parameters = JSON.load("2017-10-31-gisaxs-horizontal-cut").value
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")


consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    # Convert JSON bytes back to dict
)

async def schedule_flow(input_file_uri):
    # schedule the flow
    parameters["input_uri_data"] = input_file_uri

    await _schedule(
        deployment_name="pixel-roi-horizontal-cut-tiled/horizontal-cut",
        # deployment_name="saxs-waxs-azimuthal-integration-tiled/integration-azimuthal",
        flow_run_name=input_file_uri,
        parameters=parameters,
    )

async def kafka_consumer_scheduler():

    # Read and print messages
    for message in consumer:
        print(f"Received message: {message.value}")

        #split the message to get the uri
        input_file_uri = message.value["scan_uri"]

        parameters["input_uri_data"] = input_file_uri
        logger.info(f"Scheduling flows with {input_file_uri}")

        await schedule_flow(input_file_uri)


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith("There is no current event loop in thread"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    loop = asyncio.get_event_loop()
    loop.run_until_complete(kafka_consumer_scheduler())

    # asyncio.run(schedule_flow("raw/2024_10_31 GISAXS/1_8_A0p120_A0p121_sfloat_2m"))