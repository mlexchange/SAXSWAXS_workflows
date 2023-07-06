import numpy as np

# Alternative, remeshing at:
# https://github.com/CFN-softbio/SciAnalysis/XSAnalysis/Data.py#L990


def angle_to_pix(a, sdd, pix_size):
    """Converts an angle in degree, to a pixel position."""
    return np.tan(a / 180 * np.pi) * sdd / (pix_size / 1000)


def pix_to_angle(pixels, sdd, pix_size):
    """Converts from pixel position to angle in degree."""
    return (
        np.arctan(pixels * (pix_size / 1000) / (sdd)) / np.pi * 180
    )  # wl in AA, pix_size in µm and sdd in mm
    # q in AA^-1


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
