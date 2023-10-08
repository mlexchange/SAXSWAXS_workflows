import os

import numpy as np
import prefect
import pyFAI
from conversions import (
    filter_nans,
    mask_image,
    pix_to_alpha_f,
    pix_to_theta_f,
    q_parallel,
    q_z,
)
from dotenv import load_dotenv
from file_handling import write_1d_reduction_result
from prefect import flow, task
from tiled.client import from_uri

load_dotenv()

# Initialize the Tiled server
TILED_URI = os.getenv("TILED_URI")
TILED_API_KEY = os.getenv("TILED_API_KEY")

client = from_uri(TILED_URI, api_key=TILED_API_KEY)
TILED_BASE_URI = client.uri


@task
def pixel_roi_vertical_sum(
    image,
    beamcenter_x,
    beamcenter_y,
    incident_angle,
    sdd,
    wl,
    pix_size,
    cut_half_width,
    ymin,
    ymax,
    output_unit,
):
    shape = image.shape
    cut_data = image[
        max(0, ymin) : min(shape[0], ymax + 1),
        max(0, beamcenter_x - cut_half_width) : min(
            shape[1], beamcenter_x + cut_half_width + 1
        ),
    ]

    cut_sum = np.sum(cut_data, axis=1)
    errors = np.sqrt(cut_sum)

    pix = np.arange(ymin, ymax + 1)
    if output_unit == "pixel":
        return np.column_stack((pix, cut_sum, errors))
    else:
        # Set pixel coordinates in reference to beam center
        pix = pix - beamcenter_y
        af = pix_to_alpha_f(pix, sdd, pix_size, incident_angle)
    if output_unit == "angle":
        return np.column_stack((af, cut_sum, errors))
    elif output_unit == "q":
        qz = q_z(wl, af, incident_angle)
        return np.column_stack((qz, cut_sum, errors))


@flow
def pixel_roi_vertical_sum_tiled(
    input_uri_data: str,
    input_uri_mask: str,
    beamcenter_x: int,
    beamcenter_y: int,
    incident_angle: float,
    sdd: float,
    wl: float,
    pix_size: float,
    cut_half_width: int,
    ymin: int,
    ymax: int,
    output_unit: str,
    output_key: str,
):
    data = from_uri(TILED_BASE_URI + input_uri_data)[:]
    mask = from_uri(TILED_BASE_URI + input_uri_mask)[:]

    if data.shape != mask.shape:
        mask = np.rot90(mask)

    masked_data = mask_image(data, mask)

    reduced_data = pixel_roi_vertical_sum(
        masked_data,
        beamcenter_x,
        beamcenter_y,
        incident_angle,
        sdd,
        wl,
        pix_size,
        cut_half_width,
        ymin,
        ymax,
        output_unit,
    )

    reduced_data = filter_nans(reduced_data)

    trimmed_input_uri_data = input_uri_data.replace(TILED_BASE_URI, "")
    write_1d_reduction_result(
        trimmed_input_uri_data, "pixel_roi_vertical_sum", reduced_data, output_unit
    )
    # write back to Tiled directory directly
    # client.write_array(reduced_data, key=output_key)


@task
def pixel_roi_horizontal_sum(
    image,
    beamcenter_x,
    beamcenter_y,
    incident_angle,
    sdd,
    cut_half_width,
    xmin,
    xmax,
    wl,
    pix_size,
    output_unit,
):
    """
    Extract a cut in horizontal direction on the detector with width=2*cut_half_width.
    """
    cut_range = 2 * cut_half_width
    cut = np.zeros(xmax - xmin)
    pix = np.arange(xmin, xmax) - beamcenter_x
    for i in range(0, cut_range):
        temp_cut = image[
            beamcenter_y - cut_half_width + i : beamcenter_y - cut_half_width + (i + 1),
            xmin:xmax,
        ]
        cut += temp_cut.flatten()

    errors = np.sqrt(cut)
    af = pix_to_alpha_f(beamcenter_y + cut_half_width, sdd, pix_size, incident_angle)
    tf = pix_to_theta_f(pix, sdd, pix_size)
    qp = q_parallel(wl, tf, af, incident_angle)
    if output_unit == "q":
        return np.column_stack((qp, cut, errors))
    elif output_unit == "angle":
        return np.column_stack((tf, cut, errors))
    elif output_unit == "pixel":
        return np.column_stack((pix + beamcenter_x, cut, errors))


@task
def integrate(masked_image, pos_data, bins, error_model=None):
    """
    Integrate data values over given position data and number of bins
    """

    # initialize pyFAI setting for uncertainty estimation
    if error_model == "data variance":
        VARIANCE = abs(masked_image - masked_image.mean()) ** 2
    elif error_model == "poisson":
        VARIANCE = np.ascontiguousarray(masked_image, np.float32)
    elif error_model == "None":
        VARIANCE = None
    else:
        VARIANCE = None

    # backwards compability for PyFAI version
    if hasattr(pyFAI, "histogram"):
        histogram = pyFAI.histogram
    else:
        histogram = pyFAI.ext.histogram

    # x is q, y is intensity, weight_cything
    x, y, _, unweight_cython = histogram.histogram(
        pos_data, masked_image, bins, empty=np.nan
    )

    if VARIANCE is not None:
        xx, yy, a, b = histogram.histogram(pos_data, VARIANCE, bins)
        sigma = np.sqrt(a) / np.maximum(b, 1)
    else:
        sigma = x - x

    # mask values with no intensity information due to masking
    if unweight_cython.min() == 0:
        y = np.ma.masked_array(y, unweight_cython == 0)

        # for current dpdak version, y cant be an masked array, mask -> nan
        y = np.ma.filled(y, np.nan)
        sigma = x - x

    return x, y, sigma


@prefect.flow(name="saxs_waxs_integration")
def pyfai_reduction(
    image,
    mask,
    beamcenter_x,
    beamcenter_y,
    sdd,
    wavelength,
    tilt,
    rotation,
    pix_size,
    chi0,
    chi1,
    inner_radius,
    outer_radius,
    bins,
    profile,
):
    if mask is not None:
        masked_image = mask_image(image, mask)
    else:
        masked_image = image

    bc_x = beamcenter_x
    bc_y = (
        masked_image.shape[0] - beamcenter_y
    )  # due to flipping (depends on flipping while calibrating)

    # Corrections
    # Either correction only changes intensity, never peak locations
    # geometry correction (flat to arced, changes intensity)
    # currently not implemented or applied?
    # geom_BOOL = True
    # polarization correction (factor may be different, also changes intensity)
    # pol_BOOL = True
    # (more relevant for magnetic measurements)
    # close to 1 means unpolarized
    # TODO: Make a paramater
    pol_factor = 0.99

    # from where to where: 1 to largest possible distance (in pixels)
    if inner_radius > outer_radius:
        inner_radius, outer_radius = outer_radius, inner_radius

    # pyFAI Initialization (azimuthal integrator)
    ai = pyFAI.AzimuthalIntegrator(wavelength=wavelength)

    # pyFAI usually works with a PONI file (Point of normal incedence),
    # but here we set the geometry parameters directly in the Fit2D format
    ai.setFit2D(
        directDist=sdd,
        centerX=bc_x,
        centerY=bc_y,
        tilt=tilt,
        tiltPlanRotation=rotation,
        pixelX=pix_size,
        pixelY=pix_size,
    )

    # how much is 1q in pixels (1.578664 = 1/2pi),
    # comes from definition of q:
    # real-space -> q-space conversion
    step_size = (outer_radius - inner_radius) / float(bins) / 1.578664

    # polar angles in relation to beam center
    # (used to cut out cake piece and create mask)
    chia = np.rad2deg(ai.chiArray(masked_image.shape))
    chia[chia < 0] += 360

    # qq0 to qq1 are the extremes for integration
    # (+/- step_size is needed for pyFAI pixel conventions)
    qq0 = ai.qFunction(np.array([bc_y]), np.array([bc_x + inner_radius - step_size]))
    qq1 = ai.qFunction(np.array([bc_y]), np.array([bc_x + outer_radius + step_size]))

    # complete picture in q
    qa = ai.qArray(masked_image.shape)

    # chi1 and chi0 are two angles, defining a partial arc / cake cut
    if chi1 < chi0:
        chia[(chia >= 0) & (chia <= chi1)] += 360
        if tilt == 0.0:
            selected_subset = (chia >= chi0) & (qa > qq0) & (qa < qq1)
        else:
            selected_subset = chia >= chi0
    else:
        selected_subset = (qa > qq0) & (qa < qq1) & (chia >= chi0) & (chia <= chi1)

    # use position encoding depending on integration type
    if profile == "Radial":
        pos = qa[selected_subset]
    elif profile == "Azimuthal":
        pos = chia[selected_subset]

    # normalize for polarization
    Polar_Factor = float(pol_factor)
    masked_image /= ai.polarization(masked_image.shape, Polar_Factor)

    # Actually throw away the masked indices
    masked_image = masked_image[selected_subset]

    # conversion to sterradian (3d angle, cone, used for normalization on intensity)
    solid_angles = ai.solidAngleArray(masked_image.shape)[selected_subset]
    masked_image /= solid_angles

    # Remove masked values
    valid_indices = ~np.isnan(masked_image)
    masked_image = masked_image[valid_indices]
    pos = pos[valid_indices]

    # x_F is q or chi (depending on integration type),
    # y_F is intensity and
    # sigma_F are uncertainties in the intensity
    # TODO: make error model a parameter
    x_F, y_F, sigma_F = integrate(masked_image, pos, bins, "poisson")

    return np.column_stack((x_F, y_F, sigma_F))
