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
from scipy.signal import argrelmin, find_peaks, peak_prominences

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


@task(name="baseline_removal")
def baseline_removal(x_data, y_data, baseline_removal_method="linear"):
    # Adaptive iteratively reweighted penalized least squares (airPLS) baseline.
    if baseline_removal_method == "airpls":
        baseline_removal_obj = Baseline(y_data)
        background, _ = baseline_removal_obj.airpls()
    elif baseline_removal_method == "linear_to_inflection":
        # FIt and subtract a linear baseline to the first inflection point
        first_inflection_point = argrelmin(y_data)[0]
        slope = (y_data[first_inflection_point[0]] - y_data[0]) / (
            x_data[first_inflection_point[0]] - x_data[0]
        )
        intercept = y_data[0] - slope * x_data[0]
        y_data = y_data - (x_data * slope + intercept)
    # modified polynomial (ModPoly) baseline algorithm
    elif baseline_removal_method == "modpoly":
        baseline_removal_obj = Baseline(y_data)
        background = baseline_removal_obj.modpoly()
        y_data = y_data - background
    # Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).
    elif baseline_removal_method == "snip":
        baseline_removal_obj = Baseline(y_data)
        background = baseline_removal_obj.snip()
        y_data = y_data - background
    else:
        logger = get_run_logger()
        logger.debug(
            f"Baseline removal method {baseline_removal_method} not recognized. No baseline removal applied."
        )

    return y_data


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
):
    reduced_uri = input_uri_data.replace("raw", "processed")
    parts = reduced_uri.split("/")
    reduced_uri = f"{reduced_uri}/{parts[-1]}_{reduction_type}"
    print(reduced_uri)

    q, intensity = read_reduction_tiled(
        reduced_uri,
        q_range=fit_range,
    )

    if fit_range is None:
        fit_range_min = np.min(q)
        fit_range_max = np.max(q)
    else:
        fit_range_min = fit_range[0]
        fit_range_max = fit_range[1]

    intensity = baseline_removal(q, intensity, baseline_removal_method)

    # Find most prominent peak
    peak_indices = find_peaks(intensity)[0]
    prominences = peak_prominences(intensity, peak_indices)
    # Edit this peak number
    peak_number = np.argmax(prominences[0])
    peak_location = q[peak_indices[peak_number]]
    print("Peak location", peak_location)

    parameters_fitting = {
        "input_uri_reduction": reduced_uri,
        "x_peaks": [fit_range_min, peak_location],
        "y_peaks": [
            intensity[np.argmin(abs(q - fit_range_min))],
            intensity[np.argmin(abs(q - peak_location))],
        ],
        "stddevs": [0.00001, 0.003],
        "fwhm_Gs": [0.001, 0.001],
        "fwhm_Ls": [0.001, 0.001],
        "peak_shape": "gaussian",
        "fit_range": [fit_range_min, fit_range_max],
        "baseline_removal": baseline_removal,
    }
    simple_peak_fit_tiled(**parameters_fitting)


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
