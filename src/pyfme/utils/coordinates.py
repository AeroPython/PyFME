# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Frames of Reference orientation functions
-----------------------------------------
"""

import numpy as np
from numpy import sin, cos


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
    elif not (psi_min <= psi <= psi_max):
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

    # check_theta_phi_psi_range(theta, phi, psi)

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

    # check_theta_phi_psi_range(theta, phi, psi)

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


def check_gamma_mu_chi_range(gamma, mu, chi):
    """Check gamma, mu, chi values are inside the defined range. This
    comprobation can also detect if the value of the angle is in degrees in
    some cases.
    """

    gamma_min, gamma_max = (-np.pi/2, np.pi/2)
    mu_min, mu_max = (-np.pi, np.pi)
    chi_min, chi_max = (0, 2 * np.pi)

    if not (gamma_min <= gamma <= gamma_max):
        raise ValueError('Gamma value is not inside correct range')
    elif not (mu_min <= mu <= mu_max):
        raise ValueError('Mu value is not inside correct range')
    elif not (chi_min <= chi <= chi_max):
        raise ValueError('Chi value is not inside correct range')


def wind2hor(wind_coords, gamma, mu, chi):
    """Transforms the vector coordinates in wind frame of reference to local
    horizon frame of reference.

    Parameters
    ----------
    wind_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in wind axes.
    gamma : float
        Velocity pitch (or elevation) angle (rad).
    mu : float
        Velocity bank angle (rad).
    chi : float
        Velocity yaw (or azimuth) angle (rad)

    Returns
    -------
    hor_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in local horizon axes.

    Raises
    ------
    ValueError
        If the values of the wind-horizon angles are outside the proper ranges.

    See Also
    --------
    `hor2wind` function.

    Notes
    -----
    See [1] for frame of reference definition.
    Note that in order to avoid ambiguities ranges in angles are limited to:

    * -pi/2 <= gamma <= pi/2
    * -pi <= mu <= pi
    * 0 <= chi <= 2*pi

    References
    ----------
    .. [1] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012

    """

    check_gamma_mu_chi_range(gamma, mu, chi)

    # Transformation matrix from wind to local horizon
    Lhw = np.array([
                    [cos(gamma) * cos(chi),
                     sin(mu) * sin(gamma) * sin(chi) - cos(mu) * sin(chi),
                     cos(mu) * sin(gamma) * cos(chi) + sin(mu) * sin(chi)],
                    [cos(gamma) * sin(chi),
                     sin(mu) * sin(gamma) * sin(chi) + cos(mu) * cos(chi),
                     cos(mu) * sin(gamma) * sin(chi) - sin(mu) * cos(chi)],
                    [-sin(gamma),
                     sin(mu) * cos(gamma),
                     cos(mu) * cos(gamma)]
                    ])

    hor_coords = Lhw.dot(wind_coords)

    return hor_coords


def hor2wind(hor_coords, gamma, mu, chi):
    """Transforms the vector coordinates in local horizon frame of reference
    to wind frame of reference.

    Parameters
    ----------
    hor_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in local horizon axes.
    gamma : float
        Velocity pitch (or elevation) angle (rad).
    mu : float
        Velocity bank angle (rad).
    chi : float
        Velocity yaw (or azimuth) angle (rad)

    Returns
    -------
    wind_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in wind axes.

    Raises
    ------
    ValueError
        If the values of the wind-horizon angles are outside the proper ranges.

    See Also
    --------
    `wind2hor` function.

    Notes
    -----
    See [1] for frame of reference definition.
    Note that in order to avoid ambiguities ranges in angles are limited to:
    * -pi/2 <= gamma <= pi/2
    * -pi <= mu <= pi
    * 0 <= chi <= 2*pi

    References
    ----------
    .. [1] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012

    """

    check_gamma_mu_chi_range(gamma, mu, chi)

    # Transformation matrix from local horizon to wind
    Lwh = np.array([
                    [cos(gamma) * cos(chi),
                     cos(gamma) * sin(chi),
                     -sin(gamma)],
                    [sin(mu) * sin(gamma) * sin(chi) - cos(mu) * sin(chi),
                     sin(mu) * sin(gamma) * sin(chi) + cos(mu) * cos(chi),
                     sin(mu) * cos(gamma)],
                    [cos(mu) * sin(gamma) * cos(chi) + sin(mu) * sin(chi),
                     cos(mu) * sin(gamma) * sin(chi) - sin(mu) * cos(chi),
                     cos(mu) * cos(gamma)]
                    ])

    wind_coords = Lwh.dot(hor_coords)

    return wind_coords


def check_alpha_beta_range(alpha, beta):
    """Check alpha, beta values are inside the defined range. This
    comprobation can also detect if the value of the angle is in degrees in
    some cases.
    """

    alpha_min, alpha_max = (-np.pi/2, np.pi/2)
    beta_min, beta_max = (-np.pi, np.pi)

    if not (alpha_min <= alpha <= alpha_max):
        raise ValueError('Alpha value is not inside correct range')
    elif not (beta_min <= beta <= beta_max):
        raise ValueError('Beta value is not inside correct range')


def body2wind(body_coords, alpha, beta):
    """Transforms the vector coordinates in body frame of reference to
    wind frame of reference.

    Parameters
    ----------
    body_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in body axes.
    alpha : float
        Angle of attack (rad).
    beta : float
        Sideslip angle (rad).

    Returns
    -------
    wind_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in wind axes.

    Raises
    ------
    ValueError
        If the values of the wind-body angles are outside the proper ranges.

    See Also
    --------
    `wind2body` function.

    Notes
    -----
    See [1] or [2] for frame of reference definition.
    Note that in order to avoid ambiguities ranges in angles are limited to:
    * -pi/2 <= alpha <= pi/2
    * -pi <= beta <= pi

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight," Courier Corporation,
        pp. 104-120, 2012.
    .. [2] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012

    """

    check_alpha_beta_range(alpha, beta)

    # Transformation matrix from body to wind
    Lwb = np.array([
                    [cos(alpha) * cos(beta),
                     sin(beta),
                     sin(alpha) * cos(beta)],
                    [- cos(alpha) * sin(beta),
                     cos(beta),
                     -sin(alpha) * sin(beta)],
                    [-sin(alpha),
                     0,
                     cos(alpha)]
                    ])

    wind_coords = Lwb.dot(body_coords)

    return wind_coords


def wind2body(wind_coords, alpha, beta):
    """Transforms the vector coordinates in wind frame of reference to
    body frame of reference.

    Parameters
    ----------
    wind_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in body axes.
    alpha : float
        Angle of attack (rad).
    beta : float
        Sideslip angle (rad).

    Returns
    -------
    body_coords : array_like
        3 dimensional vector with (x,y,z) coordinates in wind axes.

    Raises
    ------
    ValueError
        If the values of the wind-body angles are outside the proper ranges.

    See Also
    --------
    `body2wind` function.

    Notes
    -----
    See [1] or [2] for frame of reference definition.
    Note that in order to avoid ambiguities ranges in angles are limited to:
    * -pi/2 <= alpha <= pi/2
    * -pi <= beta <= pi

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight," Courier Corporation,
        pp. 104-120, 2012.
    .. [2] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012

    """

    check_alpha_beta_range(alpha, beta)

    # Transformation matrix from body to wind
    Lbw = np.array([
                    [cos(alpha) * cos(beta),
                     - cos(alpha) * sin(beta),
                     -sin(alpha)],
                    [sin(beta),
                     cos(beta),
                     0],
                    [sin(alpha) * cos(beta),
                     -sin(alpha) * sin(beta),
                     cos(alpha)]
                    ])

    body_coords = Lbw.dot(wind_coords)

    return body_coords
