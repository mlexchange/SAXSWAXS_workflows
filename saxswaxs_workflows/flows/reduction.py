import os

import numpy as np
from conversions import (
    degrees_to_radians,
    filter_nans,
    get_azimuthal_integrator,
    mask_image,
    pix_to_alpha_f,
    pix_to_theta_f,
    q_parallel,
    q_z,
)
from dotenv import load_dotenv
from file_handling import (
    write_1d_reduction_result_file,
    write_1d_reduction_result_tiled,
)
from prefect import flow, get_run_logger
from tiled.client import from_uri

load_dotenv()

# Initialize the Tiled server
TILED_URI = os.getenv("TILED_URI")
TILED_API_KEY = os.getenv("TILED_API_KEY")

client = from_uri(TILED_URI, api_key=TILED_API_KEY)
TILED_BASE_URI = client.uri

WRITE_TILED_DIRECTLY = os.getenv("WRITE_TILED_DIRECTLY", False)


@flow(name="vertical-sum")
def pixel_roi_vertical_sum(
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
    shape = masked_image.shape

    y_min = max(0, y_min)
    y_max = min(shape[0], y_max + 1)

    cut_data = masked_image[
        y_min:y_max,
        max(0, int(beamcenter_x) - cut_half_width) : min(
            shape[1], int(beamcenter_x) + cut_half_width + 1
        ),
    ]

    cut_sum = np.sum(cut_data, axis=1)
    errors = np.sqrt(cut_sum)

    pix = np.arange(y_min, y_max)
    if output_unit == "pixel":
        return (pix, cut_sum, errors)
    else:
        # Set pixel coordinates in reference to beam center
        pix = pix - beamcenter_y
        af = pix_to_alpha_f(pix, sample_detector_dist, pix_size, incident_angle)
    if output_unit == "angle":
        return (af, cut_sum, errors)
    elif output_unit == "q":
        qz = q_z(wavelength, af, incident_angle)
        return (qz, cut_sum, errors)


@flow(name="pixel-roi-vertical-sum-tiled")
def pixel_roi_vertical_sum_tiled(
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
        pixel_roi_vertical_sum,
        **function_parameters,
    )


@flow(name="horizontal-sum")
def pixel_roi_horizontal_sum(
    masked_image,
    beamcenter_x,
    beamcenter_y,
    incident_angle,
    sample_detector_dist,
    wavelength,
    pix_size,
    cut_half_width,
    x_min,
    x_max,
    output_unit,
):
    """
    Extract a cut in horizontal direction on the detector with width=2*cut_half_width.
    """
    shape = masked_image.shape

    x_min = max(0, x_min)
    x_max = min(shape[1], x_max + 1)

    cut_data = masked_image[
        max(0, beamcenter_y - cut_half_width) : min(
            shape[0], beamcenter_y + cut_half_width + 1
        ),
        x_min:x_max,
    ]

    cut_sum = np.sum(cut_data, axis=1)
    errors = np.sqrt(cut_sum)

    pix = np.arange(x_min, x_max + 1)

    if output_unit == "pixel":
        return (pix, cut_sum, errors)
    else:
        pix = pix - beamcenter_x
        af = pix_to_alpha_f(
            beamcenter_y,
            sample_detector_dist,
            pix_size,
            incident_angle,
        )
        tf = pix_to_theta_f(pix, sample_detector_dist, pix_size)
    if output_unit == "angle":
        return (tf, cut_sum, errors)
    elif output_unit == "q":
        qp = q_parallel(wavelength, tf, af, incident_angle)
        return (qp, cut_sum, errors)


@flow(name="pixel-roi-horizontal-sum-tiled")
def pixel_roi_horizontal_sum_tiled(
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
        pixel_roi_horizontal_sum,
        **function_parameters,
    )


@flow(name="integration-azimuthal")
def integrate1d_azimuthal(
    masked_image,
    beamcenter_x,
    beamcenter_y,
    sample_detector_dist,
    wavelength,
    pix_size,
    tilt,
    rotation,
    chi_min,
    chi_max,
    inner_radius,
    outer_radius,
    polarization_factor,
    output_unit,  # Always q for now
    num_bins,
):
    azimuthal_integrator = get_azimuthal_integrator(
        beamcenter_x,
        beamcenter_y,
        wavelength,
        sample_detector_dist,
        tilt,
        rotation,
        pix_size,
    )

    # Convert from [0, 360] degree to [-pi, pi]
    chi_min_rad = degrees_to_radians(chi_min)
    chi_max_rad = degrees_to_radians(chi_max)

    result = azimuthal_integrator.integrate1d(
        # data=masked_image,
        data=np.copy(masked_image),
        npt=num_bins,
        correctSolidAngle=True,
        error_model="poisson",
        radial_range=(inner_radius, outer_radius),
        azimuth_range=(chi_min_rad, chi_max_rad),
        polarization_factor=polarization_factor,
    )

    return result


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
    chi_min: int,
    chi_max: int,
    inner_radius: int,
    outer_radius: int,
    polarization_factor: float,
    num_bins: int,
    output_unit: str,  # Always q for now
):
    function_parameters = locals().copy()
    reduction_tiled_wrapper(
        integrate1d_azimuthal,
        **function_parameters,
    )


@flow(name="integration-radial")
def integrate1d_radial(
    masked_image,
    beamcenter_x,
    beamcenter_y,
    sample_detector_dist,
    wavelength,
    pix_size,
    tilt,
    rotation,
    chi_min,
    chi_max,
    inner_radius,
    outer_radius,
    polarization_factor,
    num_bins,
    output_unit,  # Always q for now
):
    azimuthal_integrator = get_azimuthal_integrator(
        beamcenter_x,
        beamcenter_y,
        wavelength,
        sample_detector_dist,
        tilt,
        rotation,
        pix_size,
    )

    # Convert from [0, 360] degree to [-pi, pi]
    chi_min_rad = degrees_to_radians(chi_min)
    chi_max_rad = degrees_to_radians(chi_max)

    result = azimuthal_integrator.integrate_radial(
        # Copying here due to issue with memory ownership
        # line 323, in pyFAI.ext.splitBBoxCSR.CsrIntegrator.integrate_ng
        # "ValueError: buffer source array is read-only"
        data=np.copy(masked_image),
        # data=masked_image,
        npt=num_bins,
        correctSolidAngle=True,
        radial_range=(inner_radius, outer_radius),
        azimuth_range=(chi_min_rad, chi_max_rad),
        polarization_factor=polarization_factor,
    )

    return result


@flow(name="saxs-waxs-radial-integration-tiled")
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
    chi_min: int,
    chi_max: int,
    inner_radius: int,
    outer_radius: int,
    polarization_factor: float,
    num_bins: int,
    output_unit,  # Always q for now
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
    image = from_uri(TILED_BASE_URI + input_uri_data)[:]
    mask = from_uri(TILED_BASE_URI + input_uri_mask)[:]
    logger.debug(f"Using image from {image} and mask from {mask}.")

    masked_image = mask_image(image, mask)

    # we may want to check the parameters
    # function_to_wrap_parameters = inspect.signature(function_to_wrap).parameters
    # if "image" in function_to_wrap_parameters:
    #    function_to_wrap_parameters.pop("image")

    # Pass the masked image to the reduction function
    reduced_data = function_to_wrap(masked_image, **function_parameters)

    reduced_data = filter_nans(reduced_data)

    trimmed_input_uri = input_uri_data
    if "raw/" in trimmed_input_uri:
        trimmed_input_uri = trimmed_input_uri.replace("raw/", "")

    logger.debug(
        f"Saving {function_to_wrap.name} reduction under: processed/{trimmed_input_uri}"
    )

    if not WRITE_TILED_DIRECTLY:
        write_1d_reduction_result_file(
            trimmed_input_uri,
            function_to_wrap.name,
            reduced_data,
            **function_parameters,
        )
    else:
        write_1d_reduction_result_tiled(
            client["processed"],
            input_uri_data,
            function_to_wrap.name,
            reduced_data,
            **function_parameters,
        )


parameters_radial = {
    "input_uri_data": "raw/cali_saxs_agbh_00001/lmbdp03/cali_saxs_agbh_00001",
    "input_uri_mask": "raw/masks/saxs_mask",
    "beamcenter_x": 2945,  # x-coordiante of the beam center postion in pixel
    "beamcenter_y": 900,  # y-coordiante of the beam center postion in pixel
    "sample_detector_dist": 833.8931,  # sample-detector-distance in mm
    "pix_size": 55,  # pixel size in microns
    "wavelength": 1.05,  # wavelength in Angstrom
    "chi_min": -1.2,
    "chi_max": 1.2,
    "inner_radius": 1,
    "outer_radius": 2900,
    "polarization_factor": 0.99,
    "rotation": 49.530048,  # detector rotation in degrees (Fit2D convention)
    "tilt": 1.688493,  # detector tilt in degrees (Fit2D convention)
    "num_bins": 1450,
    "output_unit": "q",
}

if __name__ == "__main__":
    integrate1d_azimuthal_tiled(**parameters_radial)
    integrate1d_radial_tiled(**parameters_radial)
