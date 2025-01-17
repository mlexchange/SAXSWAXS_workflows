import asyncio
import logging

# import re
from pathlib import Path

import watchgod
from dotenv import load_dotenv
from file_handling import PATH_TO_RAW_DATA, TILED_BASE_URI, add_scan_tiled

# from latentspace import schedule_latent_space_reduction
from prefect.blocks.system import JSON
from utils_prefect import _schedule

# import os


load_dotenv()

logger = logging.getLogger("data_watcher")
logger.setLevel("INFO")
logname = "data_watcher_p03"
logger.addHandler(logging.StreamHandler())


# # parameters = JSON.load("2017-10-17-gisaxs-horizontal-cut").value
# parameters = JSON.load("2017-10-23-gisaxs-horizontal-cut").value


async def post_file_created(dataset_path):
    logger.info(dataset_path)
    input_file_uri = add_scan_tiled(dataset_path)
    input_file_uri = input_file_uri.replace(TILED_BASE_URI, "")
    # parameters["input_uri_data"] = input_file_uri
    # logger.info(f"Scheduling flows with {input_file_uri}")
    # await _schedule(
    #     deployment_name="horizontal_cut_automatic_fit/automatic_cut_and_fit",
    #     flow_run_name=input_file_uri,
    #     parameters=parameters,
    # )
    # # also schedule latent_space_reductions
    # # await schedule_latent_space_reduction(input_file_uri)


async def watch_directory():
    logger.info(f"Watching directory {PATH_TO_RAW_DATA}")
    async for changes in watchgod.awatch(PATH_TO_RAW_DATA):
        for change in changes:
            logger.info(change)
            logger.info(f"Detected change in {change[1]}")
            if change[0] != watchgod.Change.added:
                continue
            if ".tmp" in change[1]:
                continue
            if Path(change[1]).suffix != ".gb":
                continue
            dataset_path = change[1]
            await post_file_created(dataset_path)


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
    loop.run_until_complete(watch_directory())

    # Ingest full folder
    # dataset_folder = r"/Users/saxswaxs/733data/bl733data-2017/userdata/GuX/2024_10_16/GISAXS/Randomized"
    # for filename in os.listdir(dataset_folder):
    #    dataset_path = os.path.join(dataset_folder, filename)
    #    asyncio.run(post_file_created(dataset_path))
