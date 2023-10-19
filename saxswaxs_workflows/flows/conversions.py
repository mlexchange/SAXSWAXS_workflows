import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from prefect import task

# Alternative, remeshing at:
# https://github.com/CFN-softbio/SciAnalysis/XSAnalysis/Data.py#L990


@task
def get_azimuthal_integrator(
    beamcenter_x,
    beamcenter_y,
    wavelength,
    sample_detector_dist,
    tilt,
    rotation,
    pix_size,
):
    """
    Returns an integrator object for the given Fit2d geometry.

    Parameters
    ----------
    beamcenter_x: float
        x-coordinate of the beamcenter in pixel coordinates
    beamcenter_y: float
        y-coordinate of the beamcenter in pixel coordinates
    wavelength: float
        Wavelength in Angstrom
    sample_detector_dist: float
        Distance between sample and detector in millimeter
    tilt: float
        Tilt of the detector in degrees
    rotation: float
        Rotation in the tilt plance of the detector in degrees
    pix_size:
        Pixel size of the detector in microns

    Returns
    -------
    pyFAI.azimuthalIntegrator.AzimuthalIntegrator

    """

    azimuthal_integrator = AzimuthalIntegrator()

    # pyFAI usually works with a PONI file (Point of normal incedence),
    # but here we set the geometry parameters directly in the Fit2D format
    azimuthal_integrator.setFit2D(
        directDist=sample_detector_dist,
        centerX=beamcenter_x,
        centerY=beamcenter_y,
        tilt=tilt,
        tiltPlanRotation=rotation,
        pixelX=pix_size,
        pixelY=pix_size,
        wavelength=wavelength,
    )

    return azimuthal_integrator


@task
def filter_nans(data):
    """
    Remove numpy.NaNs from the intensity value of 1d reduced data

    Parameters
    ----------
    data : tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Tuple with of arrays where the second one represents intensity

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)

    """

    # Find Nan in the intensity
    nan_indices = np.isnan(data[1])
    cleaned_data = tuple(array[~nan_indices] for array in data)
    return cleaned_data


@task
def mask_image(image, mask):
    """
    Creates a masked array from an image and a mask, setting masked positions to NaN.

    Parameters
    ----------
    image : numpy.ndarray
        Input (detector) image
    mask : sequence or numpy.ndarray
        Mask. True indicates a masked (i.e. invalid) data.

    Returns
    -------
    numpy.ma.MaskedArray

    """
    if image.shape != mask.shape:
        mask = np.rot90(mask)

    masked_image = np.ma.masked_array(image, mask, dtype="float32", fill_value=np.nan)
    return masked_image


def degrees_to_radians(value):
    """
    Maps from an angle value in degrees [0,360] to [-pi, pi]
    """
    if value > 180:
        value -= 360
    return value * np.pi / 180


def angle_to_pix(a, sdd, pix_size):
    """Converts an angle in degree, to a length in pixel space.

    Parameters
    ----------
    a : float or numpy.ndarray
        angle in degree
    sdd : float
        sample-detector-distance in mm
    pix_size: float
        size of one pixel in µm

    Returns
    -------
    float or numpy.ndarray
    """
    return np.tan(a / 180 * np.pi) * sdd / (pix_size / 1000)


def pix_to_angle(pixels, sdd, pix_size):
    """Converts from length in pixel space to angle in degree.

    Parameters
    ----------
    pixels : float or numpy.ndarray
        pixel coordinate(s)
    sdd : float
        sample-detector-distance in mm
    pix_size: float
        size of one pixel in µm

    Returns
    -------
    float or numpy.ndarray
    """
    return np.arctan(pixels * (pix_size / 1000) / (sdd)) / np.pi * 180


def pix_to_alpha_f(pixels, sdd, pix_size, a_i):
    return pix_to_angle(pixels, sdd, pix_size) - a_i


def pix_to_theta_f(pixels, sdd, pix_size):
    return pix_to_angle(pixels, sdd, pix_size)


def q_z(wl, a_f, a_i):
    return 2 * np.pi / wl * (np.sin(a_f / 180 * np.pi) + np.sin(a_i / 180 * np.pi))


def q_y(wl, a_f, t_f):
    return 2 * np.pi / wl * np.sin(t_f / 180 * np.pi) * np.cos(a_f / 180 * np.pi)


def q_x(wl, t_f, a_f, a_i):
    return (
        2
        * np.pi
        / wl
        * (
            np.cos(t_f / 180 * np.pi) * np.cos(a_f / 180 * np.pi)
            - np.cos(a_i / 180 * np.pi)
        )
    )


def q_parallel(wl, t_f, a_f, a_i):
    qy = q_y(wl, a_f, t_f)
    qx = q_x(wl, t_f, a_f, a_i)
    return np.sqrt(qx**2 + qy**2)


def qp_to_pix(q, wl, a_f, a_i, sdd, pix_size):
    t_f = (
        180
        / np.pi
        * np.arccos(
            (
                (
                    4 * np.pi**2
                    - wl**2 * q**2
                    + 2
                    * np.pi**2
                    * (np.cos(a_f * np.pi / 90) + np.cos(a_i * np.pi / 90))
                )
                * 1
                / (np.cos(a_f * np.pi / 180))
                * 1
                / (np.cos(a_i * np.pi / 180))
            )
            / (8 * np.pi**2)
        )
    )
    pix_y = angle_to_pix(t_f, sdd, pix_size)
    return pix_y
