import inspect
import json

import numpy as np
import zmq
from saxswaxs_workflows.reduction.conversions import (
    filter_nans,
    mask_image,
)
from file_handling import (
    open_cbf,
    open_mask,
    read_array_tiled,
    write_1d_reduction_result,
    write_1d_reduction_result_files_file_only,
)

from prefect import flow, get_run_logger

from ..reduction import reduce

host = "127.0.0.1"
port = "5001"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://{}:{}".format(host, port))


@flow(name="vertical-cut")
def pixel_roi_vertical_cut(
    masked_image,
    beamcenter_x,
    beamcenter_y,
    incident_angle,
    sample_detector_dist,
    wavelength,
    pix_size,
    cut_half_width,
    y_min,
    y_max,
    output_unit,
):
    params = locals().copy()
    return reduce.pixel_roi_vertical_cut(**params)


@flow(name="pixel-roi-vertical-cut-tiled")
def pixel_roi_vertical_cut_tiled(
    input_uri_data: str,
    input_uri_mask: str,
    beamcenter_x: float,
    beamcenter_y: float,
    incident_angle: float,
    sample_detector_dist: float,
    wavelength: float,
    pix_size: float,
    cut_half_width: int,
    y_min: int,
    y_max: int,
    output_unit: str,
):
    function_parameters = locals().copy()
    reduction_tiled_wrapper(
        pixel_roi_vertical_cut,
        **function_parameters,
    )


@flow(name="horizontal-cut")
def pixel_roi_horizontal_cut(
    masked_image,
    beamcenter_x,
    beamcenter_y,
    incident_angle,
    sample_detector_dist,
    wavelength,
    pix_size,
    cut_half_width,
    cut_pos_y,
    x_min,
    x_max,
    output_unit,
):
    """
    Extract a cut in horizontal direction on the detector with width=2*cut_half_width.
    """
    param = locals().copy()
    return reduce.pixel_roi_horizontal_cut(**param)


@flow(name="pixel-roi-horizontal-cut-tiled")
def pixel_roi_horizontal_cut_tiled(
    input_uri_data: str,
    input_uri_mask: str,
    beamcenter_x: float,
    beamcenter_y: float,
    incident_angle: float,
    sample_detector_dist: float,
    wavelength: float,
    pix_size: float,
    cut_half_width: int,
    cut_pos_y: int,
    x_min: int,
    x_max: int,
    output_unit: str,
):
    function_parameters = locals().copy()
    reduction_tiled_wrapper(
        pixel_roi_horizontal_cut,
        **function_parameters,
    )


@flow(name="integration-azimuthal")
def integrate1d_azimuthal(
    image,
    mask,
    beamcenter_x,
    beamcenter_y,
    sample_detector_dist,
    wavelength,
    pix_size,
    tilt,
    rotation,
    polarization_factor,
    num_bins,
    chi_min,
    chi_max,
    inner_radius,
    outer_radius,
    output_unit,  # Always q for now
):
    params = locals().copy()
    return reduce.integrate1d_azimuthal(**params)


@flow(name="saxs-waxs-azimuthal-integration-tiled")
def integrate1d_azimuthal_tiled(
    input_uri_data: str,
    input_uri_mask: str,
    beamcenter_x: float,
    beamcenter_y: float,
    sample_detector_dist: float,
    wavelength: float,
    pix_size: int,
    tilt: float,
    rotation: float,
    polarization_factor: float,
    num_bins: int,
    chi_min: int,
    chi_max: int,
    inner_radius: int,
    outer_radius: int,
    output_unit: str,  # Always q for now
):
    function_parameters = locals().copy()
    reduction_tiled_wrapper(
        integrate1d_azimuthal,
        **function_parameters,
    )


@flow(name="saxs-waxs-azimuthal-integration-files")
def integrate1d_azimuthal_files(
    input_file_data: str,
    input_file_mask: str,
    beamcenter_x: float,
    beamcenter_y: float,
    sample_detector_dist: float,
    wavelength: float,
    pix_size: int,
    tilt: float,
    rotation: float,
    polarization_factor: float,
    num_bins: int,
    chi_min: int,
    chi_max: int,
    inner_radius: int,
    outer_radius: int,
    output_unit: str,  # Always q for now
):
    function_parameters = locals().copy()
    return reduction_files_wrapper(
        integrate1d_azimuthal,
        **function_parameters,
    )


@flow(name="integration-radial")
def integrate1d_radial(
    image,
    mask,
    beamcenter_x,
    beamcenter_y,
    sample_detector_dist,
    wavelength,
    pix_size,
    tilt,
    rotation,
    polarization_factor,
    num_bins,
    chi_min,
    chi_max,
    inner_radius,
    outer_radius,
    output_unit,  # Always q for now
):
    params = locals().copy()
    return reduce.integrate1d_radial(**params)


@flow(name="saxs-waxs-integration-radial-tiled")
def integrate1d_radial_tiled(
    input_uri_data: str,
    input_uri_mask: str,
    beamcenter_x: float,
    beamcenter_y: float,
    sample_detector_dist: float,
    wavelength: float,
    pix_size: int,
    tilt: float,
    rotation: float,
    polarization_factor: float,
    chi_min: int,
    chi_max: int,
    inner_radius: int,
    outer_radius: int,
    num_bins: int,
    output_unit: str,
):
    function_parameters = locals().copy()
    reduction_tiled_wrapper(
        integrate1d_radial,
        **function_parameters,
    )


def reduction_tiled_wrapper(
    function_to_wrap,
    **function_parameters,
):
    logger = get_run_logger()

    input_uri_data = function_parameters["input_uri_data"]
    function_parameters.pop("input_uri_data")
    input_uri_mask = function_parameters["input_uri_mask"]
    function_parameters.pop("input_uri_mask")

    # Retrieve data from Tiled
    image = read_array_tiled(input_uri_data)
    mask = read_array_tiled(input_uri_mask)

    # Mask is rotated & flipped
    if mask.shape[0] == image.shape[1] and mask.shape[1] == image.shape[0]:
        mask = np.flipud(mask)
        mask = np.rot90(mask, 3)

    logger.debug(f"Using image from {image} and mask from {mask}.")

    # we may want to check the parameters
    function_to_wrap_parameters = inspect.signature(function_to_wrap).parameters
    if "masked_image" in function_to_wrap_parameters:
        masked_image = mask_image(image, mask)
        # Pass the masked image to the reduction function
        reduced_data = function_to_wrap(masked_image, **function_parameters)
    else:
        reduced_data = function_to_wrap(image, mask, **function_parameters)

    reduced_data = filter_nans(reduced_data)

    logger.debug(f"Saving {function_to_wrap.name} reduction for: {input_uri_data}")

    result_uri = write_1d_reduction_result(
        input_uri_data,
        function_to_wrap.name,
        reduced_data,
        **function_parameters,
    )
    logger = get_run_logger()
    logger.info(f"Sending message about reduction finishing for {result_uri}")
    notification_new_reduction = {"reduction_uri": result_uri}
    # Send a message over ZMQ to someone reacting to reduction outputs
    socket.send(json.dumps(notification_new_reduction).encode())


def reduction_files_wrapper(
    function_to_wrap,
    **function_parameters,
):
    logger = get_run_logger()

    input_file_data = function_parameters["input_file_data"]
    function_parameters.pop("input_file_data")
    input_file_mask = function_parameters["input_file_mask"]
    function_parameters.pop("input_file_mask")

    # Retrieve data from Tiled
    image = open_cbf(input_file_data)
    mask = open_mask(input_file_mask)

    # Mask is rotated & flipped
    if mask.shape[0] == image.shape[1] and mask.shape[1] == image.shape[0]:
        mask = np.flipud(mask)
        mask = np.rot90(mask, 3)

    logger.debug(f"Using image from {image} and mask from {mask}.")

    # we may want to check the parameters
    function_to_wrap_parameters = inspect.signature(function_to_wrap).parameters
    if "masked_image" in function_to_wrap_parameters:
        masked_image = mask_image(image, mask)
        # Pass the masked image to the reduction function
        reduced_data = function_to_wrap(masked_image, **function_parameters)
    else:
        reduced_data = function_to_wrap(image, mask, **function_parameters)

    reduced_data = filter_nans(reduced_data)

    logger.debug(f"Saving {function_to_wrap.name} reduction for: {input_file_data}")

    return write_1d_reduction_result_files_file_only(
        input_file_data,
        function_to_wrap.name,
        reduced_data,
        **function_parameters,
    )


if __name__ == "__main__":
    parameters_azimuthal = {
        "input_uri_data": r"/raw/AgB_2024_03_25_10s_2m",
        "input_uri_mask": r"/processed/masks/AgB_2024_03_27_30s_lo_2m_mask/M_ROIMask",
        "beamcenter_x": 317.8,
        "beamcenter_y": 1245.28,
        "wavelength": 1.2398,
        "polarization_factor": 0.99,
        "sample_detector_dist": 274.83,
        "pix_size": 172,
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
    # integrate1d_azimuthal_tiled(**parameters_azimuthal)
    parameters_horizontal = {
        "input_uri_data": "raw/ALS-S2VP42/218_A0p160_A0p160_sfloat_2m",
        "input_uri_mask": "processed/masks/ALS_BCP_Mixing_inverted",
        "beamcenter_x": 670.976,
        "beamcenter_y": 1180.42,
        "incident_angle": 0.16,
        "sample_detector_dist": 3513.21,
        "wavelength": 1.2398,
        "pix_size": 172,
        "cut_half_width": 10,
        "cut_pos_y": 1180 - 105,
        "x_min": 670 - 250,
        "x_max": 670 + 250,
        "output_unit": "q",
    }
    pixel_roi_horizontal_cut_tiled(**parameters_horizontal)

if __name__ == "__main__":
    # integrate1d_azimuthal_tiled(**parameters_azimuthal)
    parameters_horizontal = {
        "input_uri_data": "raw/ALS-S2VP42/218_A0p160_A0p160_sfloat_2m",
        "input_uri_mask": "processed/masks/ALS_BCP_Mixing_inverted",
        "beamcenter_x": 670.976,
        "beamcenter_y": 1180.42,
        "incident_angle": 0.16,
        "sample_detector_dist": 3513.21,
        "wavelength": 1.2398,
        "pix_size": 172,
        "cut_half_width": 10,
        "cut_pos_y": 1180 - 105,
        "x_min": 670 - 250,
        "x_max": 670 + 250,
        "output_unit": "q",
    }
    pixel_roi_horizontal_cut_tiled(**parameters_horizontal)
