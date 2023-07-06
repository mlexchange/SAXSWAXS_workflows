import fabio
import numpy as np
import prefect
import pyFAI
from calibration import pix_to_alpha_f, pix_to_theta_f, q_parallel, q_z


def vert_cut(
    image, xpos, ypos, a_i, cut_half_width, ymin, ymax, sdd, wl, pix_size, unit
):
    cut_range = 2 * cut_half_width
    cut = np.zeros(ymax - ymin)
    pix = np.arange(ymin, ymax) - ypos
    for i in range(0, cut_range):
        temp_cut = image[
            ymin:ymax, xpos - cut_half_width + i : xpos - cut_half_width + (i + 1)
        ]
        cut += temp_cut.flatten()
    errors = np.sqrt(cut)
    af = pix_to_alpha_f(pix, sdd, pix_size, a_i)
    qz = q_z(wl, af, a_i)
    if unit == "q":
        return np.column_stack((qz, cut, errors))
    elif unit == "angle":
        return np.column_stack((af, cut, errors))
    elif unit == "pixel":
        return np.column_stack((pix + ypos, cut, errors))


# To extract a cut in horizontal direction on the detector with width=2*cut_half_width
def horz_cut(
    image, xpos, ypos, a_i, cut_half_width, xmin, xmax, sdd, wl, pix_size, unit
):
    cut_range = 2 * cut_half_width
    cut = np.zeros(xmax - xmin)
    pix = np.arange(xmin, xmax) - xpos
    for i in range(0, cut_range):
        temp_cut = image[
            ypos - cut_half_width + i : ypos - cut_half_width + (i + 1), xmin:xmax
        ]
        cut += temp_cut.flatten()
    errors = np.sqrt(cut)
    af = pix_to_alpha_f(ypos + cut_half_width, sdd, pix_size, a_i)
    tf = pix_to_theta_f(pix, sdd, pix_size)
    qp = q_parallel(wl, tf, af, a_i)
    if unit == "q":
        return np.column_stack((qp, cut, errors))
    elif unit == "angle":
        return np.column_stack((tf, cut, errors))
    elif unit == "pixel":
        return np.column_stack((pix + xpos, cut, errors))


@prefect.task
def integrate(masked_image, pos_data, mask, bins, solid_angles=None, error_model=None):
    """
    masked_image is in q-space
    """
    # Actually throw away the masked indices
    masked_image = masked_image[mask]

    if solid_angles is not None:
        masked_image /= solid_angles

    old_masked_image = list(masked_image[:])

    # additional mask for NaNs
    deleteList = []
    for i in range(len(pos_data)):
        if np.isnan(old_masked_image[i]):
            deleteList.append(False)
        else:
            deleteList.append(True)

    masked_image = masked_image[np.array(deleteList)]
    pos_data = pos_data[np.array(deleteList)]

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
    x, y, weight_cython, unweight_cython = histogram.histogram(
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
    mask_path,
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
    if mask_path == "None":
        masked_image = image
    else:
        mask = fabio.open(mask_path).data
        masked_image = np.ma.masked_array(image, mask)
        masked_image = masked_image.astype("float32")
        masked_image[masked_image.mask] = np.nan

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
    chia = ai.chiArray(masked_image.shape)
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
            mask = (chia >= chi0) & (qa > qq0) & (qa < qq1)
        else:
            mask = chia >= chi0
    else:
        mask = (qa > qq0) & (qa < qq1) & (chia >= chi0) & (chia <= chi1)

    # use position encoding depending on integration type
    if profile == "Radial":
        pos = qa[mask]
    elif profile == "Azimuthal":
        pos = chia[mask]

    # conversion to sterradian (3d angle, cone, used for normalization on intensity)
    solid_angles = ai.solidAngleArray(masked_image.shape)[mask]

    # normalize for polarization
    Polar_Factor = float(pol_factor)
    masked_image /= ai.polarization(masked_image.shape, Polar_Factor)

    # x_F is q or chi (depending on integration type),
    # y_F is intensity and
    # sigma_F are uncertainties in the intensity
    # TODO: make error model a parameter
    x_F, y_F, sigma_F = integrate(
        masked_image, pos, mask, bins, solid_angles, "poisson"
    )

    return np.column_stack((x_F, y_F, sigma_F))
