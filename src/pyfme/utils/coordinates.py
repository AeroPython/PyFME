# -*- coding: utf-8 -*-
"""
Frames of Reference orientation functions
"""

import numpy as np

# Reassing for readability
sin = np.sin
cos = np.cos


def check_theta_phi_psi_range(theta, phi, psi):
    """Check theta, phi, psi values are inside the defined range. This
    comprobation can also detect if the value of the angle is in degrees in
    some cases.
    """

    theta_min, theta_max = (-np.pi/2, np.pi/2)
    phi_min, phi_max = (-np.pi, np.pi)
    psi_min, psi_max = (0, 2 * np.pi)

    if not (theta_min <= theta <= theta_max):
        raise ValueError('Theta value is not inside correct range')
    elif not (phi_min <= phi <= phi_max):
        raise ValueError('Phi value is not inside correct range')
    elif not (psi_min <= phi <= psi_max):
        raise ValueError('Psi value is not inside correct range')



def body2hor(body_coords, theta, phi, psi):
    """Transforms the vector coordinates in body frame of reference to local
    horizon frame of reference.

    Parameters
    ----------
    body_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in body axes.
    theta : float
        Pitch (or elevation) angle (rad).
    phi : float
        Bank angle (rad).
    psi : float
        Yaw (or azimuth) angle (rad)

    Returns
    -------
    hor_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in local horizon axes.

    Raises
    ------
    ValueError
        If the values of the euler angles are outside the proper ranges.

    See Also
    --------
    `hor2body` function.

    Notes
    -----
    See [1] or [2] for frame of reference definition.
    Note that in order to avoid ambiguities ranges in angles are limited to:

    * -pi/2 <= theta <= pi/2
    * -pi <= phi <= pi
    * 0 <= psi <= 2*pi

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight," Courier Corporation,
        pp. 104-120, 2012.
    .. [2] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012
    """

    check_theta_phi_psi_range(theta, phi, psi)

    # Transformation matrix from body to local horizon
    Lhb = np.array([
                    [cos(theta) * cos(psi),
                     sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                     cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)],
                    [cos(theta) * sin(psi),
                     sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                     cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)],
                    [- sin(theta),
                     sin(phi) * cos(theta),
                     cos(phi) * cos(theta)]
                    ])

    hor_coords = Lhb.dot(body_coords)

    return hor_coords


def hor2body(hor_coords, theta, phi, psi):
    """Transforms the vector coordinates in local horizon frame of reference
    to body frame of reference.

    Parameters
    ----------
    hor_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in local horizon axes.
    theta : float
        Pitch (or elevation) angle (rad).
    phi : float
        Bank angle (rad).
    psi : float
        Yaw (or azimuth) angle (rad)

    Returns
    -------
    body_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in body axes.

    Raises
    ------
    ValueError
        If the values of the euler angles are outside the proper ranges.

    See Also
    --------
    `body2hor` function.

    Notes
    -----
    See [1] or [2] for frame of reference definition.
    Note that in order to avoid ambiguities ranges in angles are limited to:

    * -pi/2 <= theta <= pi/2
    * -pi <= phi <= pi
    * 0 <= psi <= 2*pi

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight," Courier Corporation,
        pp. 104-120, 2012.
    .. [2] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012
    """

    check_theta_phi_psi_range(theta, phi, psi)

    # Transformation matrix local horizon to body
    Lbh = np.array([
                    [cos(theta) * cos(psi),
                     cos(theta) * sin(psi),
                     - sin(theta)],
                    [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                     sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                     sin(phi) * cos(theta)],
                    [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                     cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
                     cos(phi) * cos(theta)]
                    ])

    body_coords = Lbh.dot(hor_coords)

    return body_coords
