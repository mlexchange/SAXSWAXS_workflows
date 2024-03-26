import inspect

from curve_fitting import simple_peak_fit_files
from reduction import integrate1d_azimuthal_files

from prefect import flow


@flow(name="reduction_and_fit")
def two_step_pipeline(
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
    chi_min: int,
    chi_max: int,
    inner_radius: int,
    outer_radius: int,
    num_bins: int,
    output_unit: str,
    x_peaks,
    y_peaks,
    stddevs,
    fwhm_Gs,
    fwhm_Ls,
    peak_shape,
):
    parameters = locals().copy()

    parameters_reduction = dict()
    required_parameters_reduction = inspect.signature(
        integrate1d_azimuthal_files
    ).parameters
    parameters_fitting = dict()
    required_parameters_fitting = inspect.signature(simple_peak_fit_files).parameters

    for param in parameters:
        if param in required_parameters_reduction:
            parameters_reduction[param] = parameters[param]
        if param in required_parameters_fitting:
            parameters_fitting[param] = parameters[param]

    reduction_result_file_name = integrate1d_azimuthal_files(**parameters_reduction)
    parameters_fitting["input_file_reduction"] = reduction_result_file_name

    simple_peak_fit_files(**parameters_fitting)


if __name__ == "__main__":
    parameters = {
        "input_file_data": r"Y:....\bs_pksample_c_gpcam_test_00022_00001.cbf",
        "input_file_mask": r"Y:\p03\2023\data\11019119\processed\masks\saxs_mask.tif",
        "beamcenter_x": 759,  # x-coordiante of the beam center postion in pixel
        "beamcenter_y": 1416,  # y-coordiante of the beam center postion in pixel
        "sample_detector_dist": 4248.41,  # sample-detector-distance in mm
        "pix_size": 172,  # pixel size in microns
        "wavelength": 1.044,  # wavelength in Angstrom
        "chi_min": -180,
        "chi_max": 180,
        "inner_radius": 1,
        "outer_radius": 2900,
        "polarization_factor": 0.99,
        "rotation": 0.0,  # detector rotation in degrees (Fit2D convention)
        "tilt": 0.0,  # detector tilt in degrees (Fit2D convention)
        "num_bins": 800,
        "output_unit": "q",
        "x_peaks": [0],
        "y_peaks": [1],
        "stddevs": [0.01],
        "fwhm_Gs": [0.01],
        "fwhm_Ls": [0.01],
        "peak_shape": "gaussian",
    }
    two_step_pipeline(**parameters)
