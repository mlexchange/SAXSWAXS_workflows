import os

import fabio
import h5py
import numpy as np

# from dotenv import load_dotenv
from tiled.client import from_uri

from prefect import task

# load_dotenv()

PATH_TO_RESULTS = os.path.join(os.getenv("PATH_TO_DATA"), "processed")
if not os.path.isdir(PATH_TO_RESULTS):
    os.mkdir(PATH_TO_RESULTS)

# Initialize the Tiled server
TILED_URI = os.getenv("TILED_URI")
TILED_API_KEY = os.getenv("TILED_API_KEY")

client = from_uri(TILED_URI, api_key=TILED_API_KEY)
TILED_BASE_URI = client.uri

WRITE_TILED_DIRECTLY = os.getenv("WRITE_TILED_DIRECTLY", False)


@task
def open_pil_image(pil_path, sequence_nr):
    return np.flipud(
        fabio.open(
            os.path.join(
                pil_path,
                "embl_2m",
                os.path.basename(pil_path) + "_" + str(sequence_nr).zfill(5) + ".cbf",
            ).data
        )
    )


@task
def open_lambda_image(lambda_path, sequence_nr):
    nexus_files = []
    for file in os.listdir(lambda_path):
        t = file.find(".nxs")
        if t == -1:
            pass
        else:
            nexus_files.append(file)

    module_list = []
    for file in nexus_files:
        module_list.append(int(file.split(".")[0][-2:]))
        dataset = [0]
        dataset_path = "entry/instrument/detector/data"
        ff_path = "entry/instrument/detector/flatfield"
        mask_path = "entry/instrument/detector/pixel_mask"
        trans_path = "entry/instrument/detector/translation/distance"
        size_x = 0
        size_y = 0
        lmbd_x = 516
        lmbd_y = 1556
        trans_set = {}
    dataset = [0]
    ff = [1]
    mask = [1]

    for md in range(0, len(module_list)):
        module_set = h5py.File(os.path.join(lambda_path, nexus_files[md]), "r")
        trans_set[md] = []
        trans_set[md].append(list(module_set.get(trans_path))[1])  # reverse x and y
        trans_set[md].append(list(module_set.get(trans_path))[0])
        pos_x = lmbd_x + trans_set[md][0]
        pos_y = lmbd_y + trans_set[md][1]
        size_x = (
            int(pos_x) if pos_x > size_x else int(size_x)
        )  # reform if module increase size
        size_y = int(pos_y) if pos_y > size_y else int(size_y)
        dataset.append(md)
        dataset[md] = np.array(
            module_set.get(dataset_path)[sequence_nr], dtype=np.float32
        )
        ff.append(md)
        ff[md] = 1
        ff[md] = np.array(module_set.get(ff_path)[0], dtype=np.float32)
        mask.append(md)
        mask[md] = 1
        mask[md] = np.array(module_set.get(mask_path)[0], dtype=np.float32)
        mask[md] = np.where(mask[md] > 2147483600, 0, mask[md])
        mask[md] = np.where(mask[md] == 0, 1, np.nan)
        module_set.close()
        dataset[md] *= ff[md] * mask[md]

    myimage = np.empty((size_x, size_y), dtype=np.float32)
    myimage[:] = np.NaN

    for md in range(0, len(module_list)):
        trans_x = int(trans_set[md][0])
        trans_y = int(trans_set[md][1])
        myimage[trans_x : trans_x + lmbd_x, trans_y : trans_y + lmbd_y] = np.array(
            dataset[md], dtype=np.float32
        )

    return np.flipud(np.rot90(myimage))


@task
def get_ioni_diode_from_fio(path):
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    marker = []
    header = []
    for line in lines:
        t = line.find("Col")
        ioni_string = line.find("VFC01")
        diode_string = line.find("VFC02")
        i += 1
        if t == -1:
            pass
        else:
            marker.append(i)
            header.append(line)
            if ioni_string == -1:
                pass
            else:
                ioni_col = int(line[5:6]) - 1
            if diode_string == -1:
                pass
            else:
                diode_col = int(line[5:6]) - 1

    ioni = np.genfromtxt(path, skip_header=marker[-1], usecols=ioni_col)
    diode = np.genfromtxt(path, skip_header=marker[-1], usecols=diode_col)

    res = np.column_stack((ioni, diode))

    return res


@task
def open_mask(mask_path):
    return fabio.open(mask_path).data


@task
def read_array_tiled(array_uri):
    return from_uri(TILED_BASE_URI + array_uri)[:]


# This should be following a standard
@task
def write_1d_reduction_result_file(
    trimmed_input_uri, result_type, data, **function_parameters
):
    current_folder = PATH_TO_RESULTS
    parts = trimmed_input_uri.split("/")
    for part in parts:
        current_folder = os.path.join(current_folder, part)
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)

    output_file_path = os.path.join(current_folder, f"{parts[-1]}_{result_type}.h5")

    output_file = h5py.File(output_file_path, "w")

    # default: q
    output_unit = "q"
    for key, value in function_parameters.items():
        if key == "output_unit":
            output_unit = value
        output_file.attrs[key] = value

    output_file.create_dataset(output_unit, data=data[0])
    output_file.create_dataset("intensity", data=data[1])
    if len(data) > 2:
        output_file.create_dataset("errors", data=data[2])


@task
def write_1d_reduction_result_tiled(
    processed_client, trimmed_input_uri, result_type, data, **function_parameters
):
    parts = trimmed_input_uri.split("/")
    for part in parts:
        if part not in processed_client.keys():
            processed_client.create_container(part)
        processed_client = processed_client["part"]

    final_container = f"{parts[-1]}_{result_type}"

    # default: q
    output_unit = "q"
    if "output_unit" in function_parameters:
        output_unit = function_parameters["output_unit"]

    processed_client.create_container(key=final_container, meta=function_parameters)
    processed_client = processed_client[final_container]
    processed_client.create_array(key=output_unit, array=data[0])
    processed_client.create_array(key="intensity", array=data[1])
    if len(data) > 2:
        processed_client.create_array(key="errors", array=data[2])


def write_1d_reduction_result(
    input_uri_data, result_type, reduced_data, **function_parameters
):
    trimmed_input_uri = input_uri_data
    if "raw/" in trimmed_input_uri:
        trimmed_input_uri = trimmed_input_uri.replace("raw/", "")
    if not WRITE_TILED_DIRECTLY:
        write_1d_reduction_result_file(
            trimmed_input_uri,
            result_type,
            reduced_data,
            **function_parameters,
        )
    else:
        write_1d_reduction_result_tiled(
            client["processed"],
            trimmed_input_uri,
            result_type,
            reduced_data,
            **function_parameters,
        )
