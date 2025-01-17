import os
import pathlib
import re
from datetime import timedelta

import fabio
import h5py
import numpy as np
import pandas as pd
from BaselineRemoval import BaselineRemoval
from dotenv import load_dotenv
from prefect.tasks import task_input_hash
from tiled.adapters.csv import read_csv
from tiled.adapters.hdf5 import HDF5Adapter
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure, BuiltinDtype
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.utils import ensure_uri

from prefect import flow, get_run_logger, task

load_dotenv()

PATH_TO_RAW_DATA = os.getenv("PATH_TO_DATA")
PATH_TO_PROCESSED_DATA = os.getenv("PATH_TO_PROCESSED_DATA")

if not os.path.isdir(PATH_TO_PROCESSED_DATA):
    print(PATH_TO_PROCESSED_DATA)
    path = os.path.normpath(PATH_TO_PROCESSED_DATA)
    path.mkdir(parents=True, exist_ok=True)

# Initialize the Tiled server
TILED_URI = os.getenv("TILED_URI", "")
TILED_API_KEY = os.getenv("TILED_API_KEY")

try:
    client = from_uri(TILED_URI, api_key=TILED_API_KEY)
    TILED_BASE_URI = client.uri
except Exception as e:
    print(e)


WRITE_TILED_MODE = os.getenv("WRITE_TILED", "Files")


@task
def open_cbf(file_path):
    return fabio.open(file_path).data


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


def get_parameters_from_fio(path, parameter_names, parameter_names_columns):
    parameter_patterns = [
        (param, re.compile(param + r"\s*=\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"))
        for param in parameter_names
    ]

    path = os.path.normpath(path)
    if not os.path.exists(path):
        print("File does not exsist:", path)

    parameters = dict()
    parameter_columns = dict()

    marker = 0
    line_id = 0
    with open(path, "r") as file:
        for line in file:
            line_id += 1
            for pattern in parameter_patterns:
                match = (pattern[1]).search(line)
                if match:
                    parameters[pattern[0]] = match.group(1)
            # Check if the line contains a column header
            column_found = line.find("Col")
            if column_found == -1:
                pass
            else:
                marker = line_id
                for param in parameter_names_columns:
                    param_found = line.find(param)
                    if param_found == -1:
                        pass
                    else:
                        # Assume single digit column number
                        parameter_columns[param] = int(line[5:6]) - 1

    for param in parameter_names_columns:
        if param in parameter_columns:
            column_id = parameter_columns[param]
            param_values = np.genfromtxt(path, skip_header=marker, usecols=column_id)
            parameter_columns[param] = param_values

    return parameters, parameter_columns


@task
def open_mask(mask_path):
    return fabio.open(mask_path).data


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def read_array_tiled(array_uri):
    return from_uri(TILED_BASE_URI + array_uri)[:]


# TODO: Temporaly copied here, will be removed again
def parse_txt_accompanying_edf(filepath):
    """Pase a .txt file produced at ALS beamline 7.3.3 into a dictionary.

    Parameters
    ----------
    filepath: str or pathlib.Path
        Filepath of the .edf file.
    """
    txt_filepath = None
    if isinstance(filepath, str):
        txt_filepath = filepath.replace(".edf", ".txt")
    if isinstance(filepath, pathlib.Path):
        txt_filepath = filepath.with_suffix(".txt")

    # File does not exist, return empty dictionary
    if not os.path.isfile(txt_filepath):
        return dict()

    with open(txt_filepath, "r") as file:
        lines = file.readlines()

    # Some lines have the format
    # key: value
    # others are just values with no key
    keyless_lines = 0
    txt_params = dict()
    for line in lines:
        line_components = list(map(str.strip, line.split(":", maxsplit=1)))
        if len(line_components) >= 2:
            txt_params[line_components[0]] = line_components[1]
        else:
            if line_components[0] != "!0":
                txt_params[f"Keyless Parameter #{keyless_lines}"] = line_components[0]
                keyless_lines += 1
    return txt_params


def add_scan_tiled(scan_filepath):
    common_path = os.path.commonpath([PATH_TO_RAW_DATA, scan_filepath])
    if common_path is None:
        return None

    relative_scan_filepath = os.path.relpath(scan_filepath, PATH_TO_RAW_DATA)
    scan_container, scan = os.path.split(relative_scan_filepath)
    scan_container_parts = os.path.normpath(scan_container).split(os.sep)

    current_container_client = client["raw"]
    for part in scan_container_parts:
        if part in current_container_client:
            current_container_client = current_container_client[part]
        else:
            current_container_client = current_container_client.create_container(
                key=part
            )
    key = os.path.splitext(scan)[0]

    if key in current_container_client:
        current_container_client.delete(key)

    structure = ArrayStructure(
        data_type=BuiltinDtype.from_numpy_dtype(
            np.dtype("float32") if scan_filepath.endswith(".gb") else np.dtype("int32")
        ),
        shape=(1679, 1475),
        chunks=((1679,), (1475,)),
    )

    # if scan_filepath(".edf"):
    if scan_filepath.endswith(".edf"):
        metadata = parse_txt_accompanying_edf(scan_filepath)
    else:
        metadata = {}

    # TODO: Add metadata and spec

    scan_client = current_container_client.new(
        key=key,
        structure_family=StructureFamily.array,
        data_sources=[
            DataSource(
                management=Management.external,
                mimetype=(
                    "application/x-gb"
                    if scan_filepath.endswith(".gb")
                    else "application/x-edf"
                ),
                structure_family=StructureFamily.array,
                structure=structure,
                assets=[
                    Asset(
                        data_uri=ensure_uri(scan_filepath),
                        is_directory=False,
                        parameter="data_uri",
                    )
                ],
            ),
        ],
        metadata=metadata,
        specs=[Spec("gb") if scan_filepath.endswith(".gb") else Spec("edf")],
    )
    return scan_client.uri


# This should be following a standard
@task
def write_1d_reduction_result_file(
    trimmed_input_uri, result_type, data, **function_parameters
):
    current_folder = PATH_TO_PROCESSED_DATA
    parts = trimmed_input_uri.split("/")
    for part in parts:
        current_folder = os.path.join(current_folder, part)
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)

    output_file_path = os.path.join(current_folder, f"{parts[-1]}_{result_type}.h5")

    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exist.")
    output_file = h5py.File(output_file_path, "w")

    # default: q
    output_unit = "q"
    for key, value in function_parameters.items():
        if key == "output_unit":
            output_unit = value
        output_file.attrs[key] = value

    q = data[0]
    intensity = data[1]
    output_file.attrs["input_uri"] = TILED_BASE_URI + "raw/" + trimmed_input_uri
    output_file.attrs["max_intensity"] = np.max(intensity)
    output_file.attrs["max_intensity_q"] = q[np.argmax(intensity)]
    output_file.attrs["area_under_curve"] = np.trapz(y=intensity)

    output_file.create_dataset(output_unit, data=q)
    output_file.create_dataset("intensity", data=intensity)
    if len(data) > 2:
        output_file.create_dataset("errors", data=data[2])
    return output_file_path


@task
def add_1d_reduction_result_tiled(
    processed_client,
    trimmed_input_uri,
    result_type,
    output_file_path,
):
    metadata = {"input_uri": trimmed_input_uri}
    parts = trimmed_input_uri.split("/")
    final_container = f"{parts[-1]}_{result_type}"

    # Navigate to the final container
    parts = trimmed_input_uri.split("/")
    for part in parts:
        if part not in processed_client.keys():
            processed_client.create_container(part)
        processed_client = processed_client[part]

    # We already overwrote the file, we in principle only need to update metadata
    # (not yet possible in this version)
    if final_container in processed_client:
        return processed_client[final_container].uri

    adapter = HDF5Adapter.from_uri(ensure_uri(output_file_path))
    # This will also include the function parameters
    metadata = {**metadata, **adapter.metadata()}

    # Todo extract stucture?
    processed_client.new(
        key=final_container,
        structure_family=adapter.structure_family,
        data_sources=[
            DataSource(
                management=Management.external,
                mimetype="application/x-hdf5",
                structure_family=adapter.structure_family,
                structure=adapter.structure(),
                assets=[
                    Asset(
                        data_uri=ensure_uri(output_file_path),
                        is_directory=False,
                        parameter="data_uri",
                    )
                ],
            ),
        ],
        metadata=metadata,
        specs=adapter.specs,
    )

    return processed_client[final_container].uri


@task
def write_1d_reduction_result_tiled(
    processed_client, trimmed_input_uri, result_type, data, **function_parameters
):
    parts = trimmed_input_uri.split("/")
    for part in parts:
        if part not in processed_client.keys():
            processed_client.create_container(part)
        processed_client = processed_client[part]

    final_container = f"{parts[-1]}_{result_type}"

    # default: q
    output_unit = "q"
    if "output_unit" in function_parameters:
        output_unit = function_parameters["output_unit"]
    # If result already exists overwrite it
    if final_container in processed_client:
        # Delete all its children
        prev_container_client = processed_client[final_container]
        keys_to_delete = prev_container_client.keys()
        for key in keys_to_delete:
            prev_container_client.delete(key)
        # Finally delete the container
        processed_client.delete(final_container)

    processed_client.create_container(key=final_container, metadata=function_parameters)
    processed_client = processed_client[final_container]
    processed_client.write_array(key=output_unit, array=data[0])
    processed_client.write_array(key="intensity", array=data[1])
    if len(data) > 2:
        processed_client.write_array(key="errors", array=data[2])
    return processed_client.uri


# This should be following a standard
@task
def write_1d_reduction_result_files_file_only(
    input_file, result_type, data, **function_parameters
):
    input_file = input_file.replace("raw", "processed")
    input_file = input_file.replace(".edf", "")
    input_file = os.path.normpath(input_file)
    parts = input_file.split(os.sep)
    current_folder = ""
    for part in parts:
        if part == "":
            continue
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
    return output_file_path


@flow
def write_1d_reduction_result(
    input_uri_data, result_type, reduced_data, **function_parameters
):
    trimmed_input_uri = input_uri_data
    if "raw/" in trimmed_input_uri:
        trimmed_input_uri = trimmed_input_uri.replace("raw/", "")
    logger = get_run_logger()
    logger.info(
        "Writing 1D reduction result to " + "Tiled directly"
        if WRITE_TILED_MODE == "Tiled"
        else "through files first."
    )
    if WRITE_TILED_MODE == "Files":
        output_file_path = write_1d_reduction_result_file(
            trimmed_input_uri,
            result_type,
            reduced_data,
            **function_parameters,
        )
        processed_client = client["processed"]
        result_uri = add_1d_reduction_result_tiled(
            processed_client,
            trimmed_input_uri,
            result_type,
            output_file_path,
        )
    else:
        result_uri = write_1d_reduction_result_tiled(
            client["processed"],
            trimmed_input_uri,
            result_type,
            reduced_data,
            **function_parameters,
        )
    return result_uri


def read_reduction(input_file_path):
    input_file = h5py.File(input_file_path, "r")

    y_data = np.array(input_file["intensity"])
    x_data = np.array(input_file["q"])
    # In case the data is not sorted
    x_data, y_data = zip(*sorted(zip(x_data, y_data)))

    input_file.close()

    return x_data, y_data


def read_reduction_tiled(reduction_uri, fit_range=None, baseline_removal=True):
    reduction_client = from_uri(TILED_BASE_URI + reduction_uri)
    x_data = reduction_client["q"][:]
    y_data = reduction_client["intensity"][:]

    # Filter x_data and y_data to the fit_range
    if fit_range is not None and len(fit_range) == 2:
        within_range = (x_data >= fit_range[0]) & (x_data <= fit_range[1])
        x_data = x_data[within_range]
        y_data = y_data[within_range]

    if baseline_removal is not None:
        if baseline_removal == "zhang":
            baseline_correction_obj = BaselineRemoval(y_data)
            y_data = baseline_correction_obj.ZhangFit()
        elif baseline_removal == "modpoly":
            baseline_correction_obj = BaselineRemoval(y_data)
            y_data = baseline_correction_obj.ModPoly(2)
        elif baseline_removal == "linear":
            slope = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
            intercept = y_data[0] - slope * x_data[0]
            # Subtract line
            y_data = y_data - (x_data * slope + intercept)

    # Normalize y_data
    # y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

    return x_data, y_data


def fio_file_from_scan(input_file_path):
    """
    fio file is located in raw/online/scan_name.fio, when the given file_name is
    /raw/scan_name/embl_2m/scan_name_id/
    """
    input_file_path = os.path.dirname(input_file_path)
    input_file_path = input_file_path.replace("processed", "raw")
    pattern = re.compile(
        r"(.*[\\\/]*raw)[\\\/]*([_a-z\d]+)[\\\/]*embl_2m[\\\/]*\2_\d{5}"
    )
    match = pattern.search(input_file_path)
    if match:
        beginning_path = "Y:\\p03\\2023\\data\\11019119\\raw"
        fio_file = os.path.join(beginning_path, "online", match.group(2) + ".fio")
        return fio_file
    return None


@task
def write_fitting_fio(input_file_path, fitted_x_peaks, fitted_y_peaks, fitted_fwhms):
    fio_file = fio_file_from_scan(input_file_path)
    print("Fio-file", fio_file)
    if fio_file:
        parameters_single, parameter_columns = get_parameters_from_fio(
            fio_file,
            parameter_names=["ELLI_Y", "ELLI_Z"],
            parameter_names_columns=["VFC01", "VFC02", "POS"],
        )
        elli_z_vals = parameters_single["ELLI_Z"]
        elli_y_vals = parameters_single["ELLI_Y"]
    else:
        elli_z_vals = np.array([0])
        elli_y_vals = np.array([0])
    df = pd.DataFrame(
        {
            "x": fitted_x_peaks,
            "y": fitted_y_peaks,
            "fwhm": fitted_fwhms,
            "ELLI_Y": elli_z_vals,
            "ELLI_Z": elli_y_vals,
        }
    )

    output_file_path = input_file_path.replace(
        "integration-azimuthal.h5", "fitted_peak.csv"
    )

    df.to_csv(output_file_path, index=False)


def write_fitting_tiled(
    trimmed_input_uri, fitted_x_peaks, fitted_y_peaks, fitted_fwhms
):
    parts = trimmed_input_uri.split("/")
    trimmed_output_uri = "/".join(parts[:-1])
    output_parent_client = from_uri(
        TILED_BASE_URI + trimmed_output_uri, api_key=TILED_API_KEY
    )
    output_key = f"{parts[-2]}_fitted-peaks"
    trimmed_output_uri = "/".join([trimmed_output_uri, output_key])

    input_file_path = os.path.join(
        PATH_TO_PROCESSED_DATA, trimmed_input_uri.replace("processed/", "") + ".h5"
    )
    if not os.path.exists(input_file_path):
        print("File does not exist:", input_file_path)
    output_file_path = os.path.join(
        PATH_TO_PROCESSED_DATA,
        trimmed_output_uri.replace("processed/", "") + ".csv",
    )
    print("Writing fitted peaks to", output_file_path)

    df = pd.DataFrame(
        {
            "x": fitted_x_peaks,
            "y": fitted_y_peaks,
            "fwhm": fitted_fwhms,
        }
    )

    df.to_csv(output_file_path, index=False)

    if output_key in output_parent_client:
        return

    adapter = read_csv(ensure_uri(output_file_path))
    # This will also include the function parameters
    metadata = {"input_uri": trimmed_input_uri}
    metadata = {**metadata, **adapter.metadata()}

    output_parent_client.new(
        key=output_key,
        structure_family=adapter.structure_family,
        data_sources=[
            DataSource(
                management=Management.external,
                mimetype="text/csv",
                structure_family=adapter.structure_family,
                structure=adapter.structure(),
                assets=[
                    Asset(
                        data_uri=ensure_uri(output_file_path),
                        is_directory=False,
                        parameter="data_uri",
                    )
                ],
            ),
        ],
        metadata=metadata,
        specs=adapter.specs,
    )


def write_gp_mean(first_file_name, counter, x, y, f, gp_x, gp_y):
    output_file_path = first_file_name.replace("_00000", "_" + str(counter).zfill(5))
    output_file_path = output_file_path.replace("fitted_peak.csv", "gp_state.h5")
    output_file = h5py.File(output_file_path, "w")

    output_file.create_dataset("x", data=x)
    output_file.create_dataset("y", data=y)
    output_file.create_dataset("gp_mean", data=f)
    output_file.create_dataset("gp_x", data=gp_x)
    output_file.create_dataset("gp_y", data=gp_y)


def write_gp_mean_tiled(experiment_uri, counter, x, y, f, gp_x, gp_y, hyperparameters):
    # Find file location
    current_folder = PATH_TO_PROCESSED_DATA
    trimmed_experiment_uri = experiment_uri.replace(TILED_BASE_URI, "")
    trimmed_experiment_uri = experiment_uri.replace("processed/", "")
    parts = trimmed_experiment_uri.split("/")
    parts.append("gp")
    for part in parts:
        current_folder = os.path.join(current_folder, part)
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)

    output_file_path = os.path.join(current_folder, f"gp_posterior_mean_{counter}.h5")
    print(output_file_path)
    output_file = h5py.File(output_file_path, "w")

    output_file.create_dataset("x", data=x)
    output_file.create_dataset("y", data=y)
    output_file.create_dataset("gp_mean", data=f)
    output_file.create_dataset("gp_x", data=gp_x)
    output_file.create_dataset("gp_y", data=gp_y)
    output_file.create_dataset("gp_hyerparameters", data=hyperparameters)

    adapter = HDF5Adapter.from_uri(ensure_uri(output_file_path))
    # This will also include the function parameters
    metadata = {**adapter.metadata()}

    processed_client = from_uri(experiment_uri)
    if "gp" not in processed_client.keys():
        processed_client = processed_client.create_container("gp")
    else:
        processed_client = processed_client["gp"]

    processed_client.new(
        key=f"gp_posterior_mean_{counter}",
        structure_family=adapter.structure_family,
        data_sources=[
            DataSource(
                management=Management.external,
                mimetype="application/x-hdf5",
                structure_family=adapter.structure_family,
                structure=adapter.structure(),
                assets=[
                    Asset(
                        data_uri=ensure_uri(output_file_path),
                        is_directory=False,
                        parameter="data_uri",
                    )
                ],
            ),
        ],
        metadata=metadata,
        specs=adapter.specs,
    )


if __name__ == "__main__":
    # filepath = (
    #    r"Y:\p03\2023\data\11019119\raw\online\bs_pksample_c_gpcam_test_00001.fio"
    # )
    #
    # parameters_single, parameter_columns = get_parameters_from_fio(
    #    filepath,
    #    parameter_names=["ELLI_Z", "ELLI_Y"],
    #    parameter_names_columns=["VFC01", "VFC02", "POS"],
    # )
    # print(parameters_single)
    # print(parameter_columns)
    filepath = r"/Users/saxswaxs/733data/bl733data-2017/userdata/Hexemer/2024_03_25/NaCl_1_40/NaCl_1_40_sample_446_2m.edf"  # noqa E501
    add_scan_tiled(filepath)
