import math
import time

from astropy.modeling import fitting, models
from file_handling import read_reduction, write_fitting

from prefect import flow, task


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
            # Fix peak at 0
            g.mean.fixed = True
            sum_models = g
        else:
            g = models.Gaussian1D(amplitude=y_peak, mean=x_peak, stddev=stddev)
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
            g.x_0.fixed = True
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

    write_fitting(input_file_reduction, fitted_x_peaks, fitted_y_peaks, fitted_fwhms)


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

    simple_peak_fit_files(**parameters)
