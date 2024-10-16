import asyncio
import logging

# import os
# import re
from pathlib import Path

import watchgod
from dotenv import load_dotenv
from file_handling import PATH_TO_RAW_DATA, TILED_BASE_URI, add_scan_tiled

from prefect import get_client

load_dotenv()

logger = logging.getLogger("data_watcher")
logger.setLevel("INFO")
logname = "data_watcher_p03"
logger.addHandler(logging.StreamHandler())

parameters = {
    "input_uri_data": r"/raw/AgB/AgB_2024_03_25_10s_2m",
    "input_uri_mask": r"/processed/masks/AgB_2024_03_25_10s_2m_mask",
    "beamcenter_x": 317.8,
    "beamcenter_y": 1245.28,
    "wavelength": 1.2398,
    "sample_detector_dist": 274.83,
    "pix_size": 172,
    "chi_min": -180,
    "chi_max": 180,
    "inner_radius": 1,
    "outer_radius": 2900,
    "polarization_factor": 0.99,
    "rotation": 0.0,
    "tilt": 0.0,
    "num_bins": 1024,
    "output_unit": "q",
    # "x_peaks": [0],
    # "y_peaks": [1],
    # "stddevs": [0.01],
    # "fwhm_Gs": [0.01],
    # "fwhm_Ls": [0.01],
    # "peak_shape": "gaussian",
}


async def post_file_created(dataset_path):
    logger.info(dataset_path)
    input_file_uri = add_scan_tiled(dataset_path)
    input_file_uri = input_file_uri.replace(TILED_BASE_URI, "")
    parameters["input_uri_data"] = input_file_uri
    async with get_client() as client:
        await client.create_flow_run_from_deployment(
            deployment_id="85e74c1b-04d9-4cfe-b3b1-e667301906b2",
            parameters=(parameters),
        )


async def watch_directory():
    logger.info(f"Watching directory {PATH_TO_RAW_DATA}")
    async for changes in watchgod.awatch(PATH_TO_RAW_DATA):
        for change in changes:
            logger.info(f"Detected change in {change[1]}")
            if change[0] != watchgod.Change.added:
                continue
            if ".tmp" in change[1]:
                continue
            if Path(change[1]).suffix != ".edf":
                continue
            dataset_path = change[1]
            await post_file_created(dataset_path)


if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(watch_directory())

    # dataset_path = r"Y:\p03\2023\data\11019119\raw\bs_pksample_c_gpcam_test_00022\embl_2m\bs_pksample_c_gpcam_test_00022_00001.cbf"  # noqa: E501
    # dataset_path = r"/Users/saxswaxs/733data/bl733data-2017/userdata/Hexemer/2024_03_25/NaCl_1_40/NaCl_1_40_sample_42_2m.edf" # noqa: E501
    dataset_path = r"/Users/wiebke/Documents/Data/saxs-waxs-samples/als/2024_03_25/AgB_2024_03_25_10s_2m.edf"  # noqa: E501
    asyncio.run(post_file_created(dataset_path))
