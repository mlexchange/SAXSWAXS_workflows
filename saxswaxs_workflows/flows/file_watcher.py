import asyncio
import logging
import os
import sys

import watchgod
from dotenv import load_dotenv
from prefect import get_client

load_dotenv()

PATH_TO_DATA = os.getenv("PATH_TO_DATA")
logger = logging.getLogger("data_watcher.p03")
logger.setLevel("INFO")
logname = "data_watcher_p03.log"
logger.addHandler(logging.StreamHandler(sys.stdout))

parameters_azimuthal = {
    "input_file_data": "Y:\\p03\\2023\\data\\xxxxxxxx\\raw\\xxx\\embl_2m\\xxx.cbf",
    "input_file_mask": "Y:\\p03\\2023\\data\\xxxxxxxx\\processed\\masks\\saxs_mask.tif",
    "beamcenter_x": 759,
    "beamcenter_y": 1416,
    "sample_detector_dist": 4248.41,
    "pix_size": 172,
    "wavelength": 1.044,
    "chi_min": -180,
    "chi_max": 180,
    "inner_radius": 1,
    "outer_radius": 2900,
    "polarization_factor": 0.99,
    "rotation": 0.0,
    "tilt": 0.0,
    "num_bins": 800,
    "output_unit": "q",
}


async def post_file_created(client, dataset_path):
    logger.info(dataset_path)
    parameters_azimuthal["input_file_data"] = dataset_path
    async with get_client() as client:
        await client.create_flow_run_from_deployment(
            deployment_id="8154e634-87df-421e-b94c-25f67a374033",
            parameters=parameters_azimuthal,
        )


async def watch_directory():
    logger.info(f"Watching directory {PATH_TO_DATA}")
    client = None
    async for changes in watchgod.awatch(PATH_TO_DATA):
        for change in changes:
            logger.info(f"Detected change in {change[1]}")
            if change[0] != watchgod.Change.added:
                continue
            if ".cbf" not in change[1]:
                continue
            dataset_path = change[1]
            await post_file_created(client, dataset_path)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(watch_directory())
