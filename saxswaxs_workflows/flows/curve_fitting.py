import math
import time

import numpy as np
from astropy.modeling import fitting, models
from file_handling import (
    read_reduction,
    read_reduction_tiled,
    write_fitting_fio,
    write_fitting_tiled,
)
from pybaselines import Baseline
from scipy.signal import argrelmin, find_peaks

from prefect import flow, get_run_logger, task


@task
def curve_fitting(
    x_data, y_data, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, peak_shape="gaussian"
):
    if peak_shape == "gaussian":
        fitted_model, partial_fits = _fit_gaussian(
            x_data, y_data, x_peaks, y_peaks, stddevs
        )
    else:
        fitted_model, partial_fits = _fit_voigt(
            x_data, y_data, x_peaks, y_peaks, fwhm_Ls, fwhm_Gs
        )
    return fitted_model, partial_fits


@task
def get_fitting_params(fitted_model, peak_shape="gaussian"):
    if peak_shape == "gaussian":
        fitted_x_peaks, fitted_y_peaks, fitted_fwhm = _get_gaussian_params(fitted_model)
    else:
        fitted_x_peaks, fitted_y_peaks, fitted_fwhm = _get_voigt_params(fitted_model)
    return fitted_x_peaks, fitted_y_peaks, fitted_fwhm


def _fit_gaussian(x_data, y_data, x_peaks, y_peaks, stddevs):
    partial_fits = []
    start = time.time()
    for ii, (x_peak, y_peak, stddev) in enumerate(zip(x_peaks, y_peaks, stddevs)):
        if ii == 0:
            g = models.Gaussian1D(amplitude=y_peak, mean=x_peak, stddev=stddev)
            # Restrict amplitude to be positive
            g.amplitude.min = 0
            sum_models = g
        else:
            g = models.Gaussian1D(amplitude=y_peak, mean=x_peak, stddev=stddev)
            sum_models += g
            # Restrict amplitude to be positive
            g.amplitude.min = 0
        partial_fits.append(g)
    fitter = fitting.LevMarLSQFitter()
    try:
        fitted_model = fitter(sum_models, x_data, y_data)
    except Exception as e:
        print(f"Fitting failed due to: {e}")
        fitted_model = None
    print(f"Curve fitting done in {time.time()-start}")
    return fitted_model, partial_fits


def _fit_voigt(x_data, y_data, x_peaks, y_peaks, fwhm_Ls, fwhm_Gs):
    partial_fits = []
    start = time.time()
    for ii, (x_peak, y_peak, fwhm_L, fwhm_G) in enumerate(
        zip(x_peaks, y_peaks, fwhm_Ls, fwhm_Gs)
    ):
        if ii == 0:
            g = models.Voigt1D(
                amplitude_L=y_peak, x_0=x_peak, fwhm_L=fwhm_L, fwhm_G=fwhm_G
            )
            # Fix peak at 0
            # g.x_0.fixed = True
            g.mean.max = np.min(x_data)
            sum_models = g
        else:
            g = models.Voigt1D(
                amplitude_L=y_peak, x_0=x_peak, fwhm_L=fwhm_L, fwhm_G=fwhm_G
            )
            sum_models += g
        partial_fits.append(g)
    fitter = fitting.LevMarLSQFitter()
    try:
        fitted_model = fitter(sum_models, x_data, y_data)
    except Exception as e:
        print(f"Fitting failed due to: {e}")
        fitted_model = None
    print(f"Curve fitting done in {time.time()-start}")
    return fitted_model, partial_fits


def _get_gaussian_params(fitted_model):
    num_peaks = fitted_model.n_submodels
    fitted_x_peaks = []
    fitted_y_peaks = []
    fitted_fwhm = []
    # Update peak and FWHM information
    if fitted_model is not None:
        for ii in range(num_peaks):
            if num_peaks == 1:
                param_mean = "mean"
                param_stddev = "stddev"
            else:
                param_mean = f"mean_{ii}"
                param_stddev = f"stddev_{ii}"
            curve_mean = eval(f"fitted_model.{param_mean}.value")
            curve_stddev = eval(f"fitted_model.{param_stddev}.value")
            fitted_x_peaks.append(curve_mean)
            fitted_y_peaks.append(fitted_model(curve_mean))
            fitted_fwhm.append(2 * curve_stddev * math.sqrt(2 * math.log(2)))
    return fitted_x_peaks, fitted_y_peaks, fitted_fwhm


def _get_voigt_params(fitted_model):
    num_peaks = fitted_model.n_submodels
    fitted_x_peaks = []
    fitted_y_peaks = []
    fitted_fwhm = []
    # Update peak and FWHM information
    if fitted_model is not None:
        for ii in range(num_peaks):
            if num_peaks == 1:
                paramx0 = "x_0"
                param_L = "fwhm_L"
                param_G = "fwhm_G"
            else:
                paramx0 = f"x_0_{ii}"
                param_L = f"fwhm_L_{ii}"
                param_G = f"fwhm_G_{ii}"
            curve_x0 = eval(f"fitted_model.{paramx0}.value")
            curve_fwhm_l = eval(f"fitted_model.{param_L}.value")
            curve_fwhm_g = eval(f"fitted_model.{param_G}.value")
            fitted_x_peaks.append(curve_x0)
            fitted_y_peaks.append(fitted_model(curve_x0))
            fitted_fwhm.append(
                0.5346 * curve_fwhm_l
                + math.sqrt(0.2166 * (curve_fwhm_l**2) + (curve_fwhm_g**2))
            )
    return fitted_x_peaks, fitted_y_peaks, fitted_fwhm


def iterative_background_subtraction(intensity, shift_width=2, iterations=10000):
    """
    Perform iterative background subtraction on a 1D intensity array.

    This function creates multiple shifted versions of the intensity array over a
    specified 'shift_width', calculates their average, and then replaces any value in
    the original array that is greater than this average with the average value.
    This process is repeated for a specified number of iterations.

    Parameters
    ----------
    intensity : np.ndarray
        The raw intensity data array.
    shift_width : int, optional
        The maximum shift used for creating shifted versions of the array. For example,
        if shift_width=2, the function will create arrays shifted by -2, -1, 0, 1, and 2.
        (Default is 2.)
    iterations : int, optional
        The number of iterations to perform (default is 100000).

    Returns
    -------
    np.ndarray
        The background (or baseline) estimate obtained from the iterative subtraction.
    """
    # Create a working copy of the intensity array and add a small offset
    # to avoid potential issues with zeros or negative values.
    corrected_intensity = intensity * 1.0

    for _ in range(iterations):
        # Build a list of arrays to average: the unshifted array plus all shifts
        arrays_to_average = []

        # Create shifted arrays from -shift_width to +shift_width, excluding 0.
        for shift in range(1, shift_width + 1):
            # Shift to the left by 'shift'
            shifted_left = np.roll(corrected_intensity, shift=shift)
            # Correct the first 'shift' elements that got rolled from the end
            shifted_left[:shift] = corrected_intensity[:shift]
            arrays_to_average.append(shifted_left)

            # Shift to the right by 'shift'
            shifted_right = np.roll(corrected_intensity, shift=-shift)
            # Correct the last 'shift' elements that got rolled from the beginning
            shifted_right[-shift:] = corrected_intensity[-shift:]
            arrays_to_average.append(shifted_right)

        # Compute the mean across all shifted versions
        average_array = np.mean(arrays_to_average, axis=0)

        # Optionally preserve the first/last few points to avoid edge artifacts
        # (For example, you might do this):
        average_array[:shift_width] = corrected_intensity[:shift_width]
        average_array[-shift_width:] = corrected_intensity[-shift_width:]

        # Replace points that are above the average with the average
        mask = corrected_intensity > average_array
        corrected_intensity[mask] = average_array[mask]

    return corrected_intensity


@flow(name="baseline_removal")
def baseline_removal(x_data, y_data, baseline_removal_method="linear"):
    y_data = np.copy(y_data)
    logger = get_run_logger()
    # Adaptive iteratively reweighted penalized least squares (airPLS) baseline.
    if baseline_removal_method == "airpls":
        baseline_removal_obj = Baseline(x_data=y_data)
        background, _ = baseline_removal_obj.airpls()
    # Fit and subtract a linear baseline to the first inflection point
    elif baseline_removal_method == "linear_to_inflection":
        first_inflection_point = argrelmin(y_data)[0]
        logger.info(
            f"Found first inflection point at {x_data[first_inflection_point[0]]}."
        )
        slope = (y_data[first_inflection_point[0]] - y_data[0]) / (
            x_data[first_inflection_point[0]] - x_data[0]
        )
        intercept = y_data[0] - slope * x_data[0]
        logger.info(f"Linear baseline slope: {slope}, intercept: {intercept}.")
        background = slope * x_data + intercept
    # modified polynomial (ModPoly) baseline algorithm
    elif baseline_removal_method == "modpoly":
        baseline_removal_obj = Baseline(x_data=y_data)
        background, _ = baseline_removal_obj.modpoly()
    # Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).
    elif baseline_removal_method == "snip":
        baseline_removal_obj = Baseline(x_data=y_data)
        background, _ = baseline_removal_obj.snip()
    elif baseline_removal_method == "rolling_window":
        background = iterative_background_subtraction(y_data, shift_width=15)
    elif baseline_removal_method == "rolling_window_log":
        background = np.exp(
            iterative_background_subtraction(np.log(y_data), shift_width=15)
        )
    else:
        logger = get_run_logger()
        logger.debug(
            f"Baseline removal method {baseline_removal_method} not recognized. No baseline removal applied."
        )
        background = np.zeros_like(y_data)

    y_data -= background
    return y_data, background


@flow(name="simple_peak_fit")
def simple_peak_fit(
    x_data, y_data, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, peak_shape
):
    fitted_model, _ = curve_fitting(
        x_data, y_data, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, peak_shape
    )

    if fitted_model is not None:
        fitted_x_peaks, fitted_y_peaks, fitted_fwhms = get_fitting_params(
            fitted_model, peak_shape
        )

    return fitted_x_peaks, fitted_y_peaks, fitted_fwhms


@flow(name="simple_peak_fit_file")
def simple_peak_fit_files(
    input_file_reduction, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, peak_shape
):
    x_data, y_data = read_reduction(input_file_reduction)

    fitted_x_peaks, fitted_y_peaks, fitted_fwhms = simple_peak_fit(
        x_data, y_data, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, peak_shape
    )

    write_fitting_fio(
        input_file_reduction, fitted_x_peaks, fitted_y_peaks, fitted_fwhms
    )


@flow(name="simple_peak_fit_tiled")
def simple_peak_fit_tiled(
    input_uri_reduction,
    x_peaks,
    y_peaks,
    stddevs,
    fwhm_Gs,
    fwhm_Ls,
    peak_shape,
    fit_range=None,
):
    x_data, y_data = read_reduction_tiled(input_uri_reduction, q_range=fit_range)

    fitted_x_peaks, fitted_y_peaks, fitted_fwhms = simple_peak_fit(
        x_data, y_data, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, peak_shape
    )

    write_fitting_tiled(
        input_uri_reduction, fitted_x_peaks, fitted_y_peaks, fitted_fwhms
    )


@flow(name="automatic_peak_fit_tiled")
def automatic_peak_fit_tiled(
    input_uri_data,
    reduction_type,
    fit_range=None,
    baseline_removal_method=None,
    peak_prominence=1,
    max_num_peaks=3,
):
    logger = get_run_logger()

    reduced_uri = input_uri_data.replace("raw", "processed")
    parts = reduced_uri.split("/")
    reduced_uri = f"{reduced_uri}/{parts[-1]}_{reduction_type}"
    logger.info(f"Peak fitting {reduced_uri}")

    q, intensity = read_reduction_tiled(
        reduced_uri,
        q_range=fit_range,
    )

    # Background removal
    intensity, background = baseline_removal(q, intensity, baseline_removal_method)

    # Find all peaks in the data
    peak_indices, peak_parameters = find_peaks(intensity, prominence=peak_prominence)

    # Having given the prominence keywork to find_peaks, we can extract the prominence values
    prominences = peak_parameters["prominences"]
    right_bases = peak_parameters["right_bases"]
    left_bases = peak_parameters["left_bases"]

    num_peaks = len(peak_indices)
    if num_peaks == 0:
        logger.info("No peaks found in the specified range.")
        return

    if max_num_peaks > num_peaks:
        max_num_peaks = num_peaks
        logger.info(
            f"Only {num_peaks} peaks found, adjusting max_num_peaks to {max_num_peaks}."
        )

    # Select the peaks with the highest prominence
    sorted_peak_indices = np.argsort(prominences)[-max_num_peaks:]
    peak_indices = peak_indices[sorted_peak_indices]
    prominences = prominences[sorted_peak_indices]
    left_bases = left_bases[sorted_peak_indices]
    right_bases = right_bases[sorted_peak_indices]

    x_peaks = q[peak_indices]
    y_peaks = intensity[peak_indices]
    stddevs = np.zeros_like(x_peaks)
    fwhm_Gs = np.zeros_like(x_peaks)
    fwhm_Ls = np.zeros_like(x_peaks)

    for idx, peak_idx in enumerate(peak_indices):
        peak_width = q[right_bases[idx]] - q[left_bases[idx]]

        # Assuming the peak widths covers approximately -3/+3 standard deviations
        stddevs[idx] = peak_width / 6
        # Estimate FWHM from standard deviation
        fwhm_Gs[idx] = 2 * stddevs[idx] * math.sqrt(2 * math.log(2))
        fwhm_Ls[idx] = fwhm_Gs[idx]

    fitted_x_peaks, fitted_y_peaks, fitted_fwhms = simple_peak_fit(
        q, intensity, x_peaks, y_peaks, stddevs, fwhm_Gs, fwhm_Ls, "gaussian"
    )

    write_fitting_tiled(reduced_uri, fitted_x_peaks, fitted_y_peaks, fitted_fwhms)


if __name__ == "__main__":
    parameters = {
        "input_file_reduction": "test_integration-azimuthal.h5",
        "x_peaks": [0],
        "y_peaks": [1],
        "stddevs": [0.01],
        "fwhm_Gs": [0.01],
        "fwhm_Ls": [0.01],
        "peak_shape": "gaussian",
    }
    # simple_peak_fit_files(**parameters)

    q_target = 2 * 3.14 / (30) * 0.1
    processed_uri = (
        "processed/ALS-S2VP42/218_A0p160_A0p160_sfloat_2m/"
        + "218_A0p160_A0p160_sfloat_2m_horizontal-cut"
    )
    parameters_fitting = {
        "input_uri_reduction": processed_uri,
        "x_peaks": [0, q_target],
        "y_peaks": [1000, 1000],
        "stddevs": [0.01, 0.01],
        "fwhm_Gs": [0.01, 0.01],
        "fwhm_Ls": [0.01, 0.01],
        "peak_shape": "gaussian",
    }

    simple_peak_fit_tiled(**parameters_fitting)
