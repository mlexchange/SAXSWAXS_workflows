import json
import logging
import os
import time
from datetime import datetime

import pytz
from dotenv import load_dotenv
from tiled.client import from_uri
from utils_prefect import _schedule, schedule_prefect_flow

load_dotenv(".latentspace_env")

# Set this to the experiment uri
DATA_TILED_URI = os.getenv(
    "DEFAULT_TILED_URI", "http://127.0.0.1:8888/api/v1/metadata/raw/"
)
DATA_TILED_API_KEY = os.getenv("API_KEY")
FLOW_NAME = os.getenv("FLOW_NAME", "")
PREFECT_TAGS = ["latent-space-explorer-live"]
WRITE_DIR = os.getenv("WRITE_DIR")
RESULT_TILED_URI = os.getenv("RESULT_TILED_URI")
RESULT_TILED_API_KEY = os.getenv("RESULT_TILED_API_KEY", None)
AUTO_ARGUMENTS = os.getenv("AUTO_ARGUMENTS")
AUTO_MODEL_DIR = os.getenv("AUTO_MODEL_DIR")
PCA_MODEL_URI = os.getenv("PCA_MODEL_URI")
PCA_ARGUMENTS = os.getenv("PCA_ARGUMENTS")
TIMEZONE = os.getenv("TIMEZONE", "UTC")
PUBLISHER_PYTHON_FILE = os.getenv("PUBLISHER_PYTHON_FILE")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

with open(AUTO_ARGUMENTS, "r") as file:
    auto_params = json.load(file)
with open(PCA_ARGUMENTS, "r") as file:
    pca_params = json.load(file)

flow = {
    "flow_type": "conda",
    "params_list": [
        {
            "conda_env_name": "mlex_pytorch_autoencoders",
            "python_file_name": "/data/tanchavez/latent_space/mlex_pytorch_autoencoders/src/predict_model.py",
            "params": {
                "io_parameters": {
                    "uid_retrieve": "",
                    "data_uris": [],
                    "data_tiled_api_key": DATA_TILED_API_KEY,
                    "data_type": "tiled",
                    "root_uri": DATA_TILED_URI,
                    "output_dir": f"{WRITE_DIR}/feature_vectors",
                    "result_tiled_uri": RESULT_TILED_URI,
                    "result_tiled_api_key": RESULT_TILED_API_KEY,
                    "model_dir": AUTO_MODEL_DIR + "/last.ckpt",
                },
                "model_parameters": auto_params,
            },
        },
        {
            "conda_env_name": "mlex_dimension_reduction_pca",
            "python_file_name": "/data/tanchavez/latent_space/mlex_dimension_reduction_pca/pca_run.py",
            "params": {
                "io_parameters": {
                    "uid_retrieve": "",
                    "data_uris": [],
                    "data_tiled_api_key": DATA_TILED_API_KEY,
                    "data_type": "tiled",
                    "root_uri": DATA_TILED_URI,
                    "output_dir": f"{WRITE_DIR}/latent_vectors",
                    "result_tiled_uri": RESULT_TILED_URI,
                    "result_tiled_api_key": RESULT_TILED_API_KEY,
                    "load_model_path": PCA_MODEL_URI,
                },
                "model_parameters": pca_params,
            },
        },
        {
            "conda_env_name": "mlex_rabbitmq_publisher",
            "python_file_name": PUBLISHER_PYTHON_FILE,
            "params": {"io_parameters": {"uid_retrieve": ""}},
        },
    ],
}


def get_data_list(tiled_uri, tiled_api_key=None):
    client = from_uri(tiled_uri, api_key=tiled_api_key)
    data_list = client.keys()[10 : len(client.keys())]
    return data_list


async def schedule_latent_space_reduction(data_uri):
    logger.info(f"Sending URI {data_uri} for processing.")
    new_flow = flow.copy()
    new_flow["params_list"][0]["params"]["io_parameters"]["data_uris"] = [data_uri]
    new_flow["params_list"][1]["params"]["io_parameters"]["data_uris"] = [data_uri]
    current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y/%m/%d %H:%M:%S")
    job_name = f"Live model training for {data_uri}"
    await _schedule(
        FLOW_NAME,
        flow_run_name=f"{job_name} {current_time}",
        parameters=new_flow,
        tags=PREFECT_TAGS + ["train"],
    )


if __name__ == "__main__":
    data_list = get_data_list(DATA_TILED_URI, DATA_TILED_API_KEY)
    for data_uri in data_list:
        logger.info(f"Sending URI {data_uri} for processing.")
        new_flow = flow.copy()
        new_flow["params_list"][0]["params"]["io_parameters"]["data_uris"] = [data_uri]
        new_flow["params_list"][1]["params"]["io_parameters"]["data_uris"] = [data_uri]
        current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
            "%Y/%m/%d %H:%M:%S"
        )
        job_name = f"Live model training for {data_uri}"
        schedule_prefect_flow(
            FLOW_NAME,
            new_flow,
            flow_run_name=f"{job_name} {current_time}",
            tags=PREFECT_TAGS + ["train"],
        )
        time.sleep(10)
