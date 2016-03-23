# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Ball
----

Created on Fri Jan  8 18:53:17 2016

@author: JuanMatSa
"""

import numpy as np
from pyfme.utils.coordinates import wind2body

# Constants
mu = 1.983e-5  # kg/m/s
# Data tables
Re_list = np.array([38000, 100000, 160000, 200000, 250000, 300000, 330000,
                    350000, 375000, 400000, 500000, 800000, 200000, 4000000])
CD_full_list = np.array([0.49, 0.50, 0.51, 0.51, 0.49, 0.46, 0.39, 0.20, 0.09,
                         0.07, 0.07, 0.10, 0.15, 0.18])
Sn_list = np.array([0.00, 0.04, 0.10, 0.20, 0.40])
C_magnus_list = np.array([0.00, 0.10, 0.16, 0.23, 0.33])


def geometric_data(r=0.111):

    """Provides the value of some geometric data.

    Parameters
    ----------
    r : float
        radius(m)

    Returns
    -------
    r : float
        radius(m)
    S_circle : float
        Surface (m2)
    S_sphere : float
        Surface (m2)
    Vol : float
        Volume (m3)
    """

    S_circle = np.pi * r ** 2
    S_sphere = 4 * np.pi * r ** 2
    Vol = 4 / 3. * np.pi * r ** 3

    return r, S_circle, S_sphere, Vol


def mass_and_inertial_data(r, mass=0.440):

    """Provides the value of some mass and inertial data.

    Parameters
    ----------
    r : float
        radius (m)
    mass : float
        mass (kg)

    Returns
    -------
    inertia : float array
        diagonal array (3x3) wich elements are Ixx_b, Iyy_b, Izz_b:
    Ixx_b : float
        Moment of Inertia x-axis (Kg * m2)
    Iyy_b : float
        Moment of Inertia y-axis (Kg * m2)
    Izz_b : float
        Moment of Inertia z-axis (Kg * m2)
    """

    Ixx_b = 2 * mass * (r ** 2) / 3.
    Iyy_b = Ixx_b
    Izz_b = Ixx_b

    inertia = np.diag([Ixx_b, Iyy_b, Izz_b])

    return inertia


def check_reynolds_number(Re):
    """Reynolds number must be between 38e3 and 4e6
    Parameters
    ----------
    Re : float
        Reynolds number

    Raises
    ------
    ValueError
        If the value of the Reynolds number is outside the defined layers.
    """
    if not (Re_list[0] <= Re <= Re_list[-1]):
        raise ValueError('Reynolds number is not inside correct range')


def check_sn(Sn):
    """Effective spin number must be between 0.00 and 0.40
    Parameters
    ----------
    Sn : float
        Effective spin number

    See Also
    --------
    get_magnus_effect_forces function

    Raises
    ------
    ValueError
        If the value of the effective spin number is outside the defined
        layers.

    """
    if not (Sn_list[0] <= Sn <= Sn_list[-1]):
        raise ValueError('Effective spin number is not inside correct range')


def get_aerodynamic_forces(lin_vel, ang_vel, TAS, rho, alpha, beta,
                           magnus_effect=True):
    """Given the velocity vectors vel and ang_vel (body axes) it provides the
    forces (aerodynamic drag and Magnus effect if it exists) (body axes).
    Data for a soccer ball (smooth sphere).

    Parameters
    ----------
    lin_vel : float array
        [u, v, w] air linear velocity body-axes (m/s).
    ang_vel : float array
        [p, q, r] air angular velocity body-axes (m/s).
    TAS : float
        true air speed (m/s)
    rho : float
        air density (kg/m3).
    alpha : float
        angle of attack (rad).
    beta : float
        sideslip angle (rad).
    magnus_effect : boolean
        True if magnus effect is under consideration.

    Returns
    -------
    Total_aerodynamic_forces_body : float array
        Drag (+ magnus effect if exists) Forces vector (1x3) (body axes) (N).

    See Also
    --------
    [2]

    Notes
    -----
    Smooth ball selected. see[1]

    Reynolds vs CD_full table:

    +------------+------------+
    | Re         |CD_full_list|
    +============+============+
    | 3.8e4      |    0.49    |
    +------------+------------+
    |1.e5        |   0.51     |
    +------------+------------+
    |100000      |   0.50     |
    +------------+------------+
    |160000      |   0.51     |
    +------------+------------+
    |200000      |   0.51     |
    +------------+------------+
    |250000      |   0.49     |
    +------------+------------+
    |300000      |   0.46     |
    +------------+------------+
    |330000      |   0.39     |
    +------------+------------+
    |350000      |   0.20     |
    +------------+------------+
    |375000      |   0.09     |
    +------------+------------+
    |400000      |   0.07     |
    +------------+------------+
    |500000      |   0.07     |
    +------------+------------+
    |800000      |   0.10     |
    +------------+------------+
    |2000000     |   0.15     |
    +------------+------------+
    |4000000     |   0.18     |
    +------------+------------+

    Re : float
        Re = rho * TAS * radius / mu  # Reynolds number. values between 38e3
        and 4e6.

    References
    ----------
    .. [1] "Aerodynamics of Sports Balls" Bruce D. Jothmann, January 2007
    .. [2] "Aerodynamics of Sports Balls" Annual Review of Fluid Mechanics,
            1875.17:15 Mehta, R.D.
    """
    u, v, w = lin_vel  # components of the linear velocity
    p, q, r = ang_vel  # components of the angular velocity

    radius, A_front, _, _ = geometric_data()
    Re = rho * TAS * radius / mu  # Reynolds number
    check_reynolds_number(Re)

    # %Obtaining of Drag coefficient and Drag force
    CD_full = np.interp(Re, Re_list, CD_full_list)

    D = 0.5 * rho * TAS ** 2 * A_front * CD_full
    D_vector_body = wind2body(([-D, 0, 0]), alpha, beta)
    # %It adds or not the magnus effect, depending on the variable
    # % magnus_effect
    if magnus_effect:
        F_magnus_vector_body = get_magnus_effect_forces(lin_vel, ang_vel, TAS,
                                                        rho, radius, A_front,
                                                        alpha, beta)

        total_aerodynamic_forces_body = D_vector_body + F_magnus_vector_body
        return total_aerodynamic_forces_body
    else:
        total_aerodynamic_forces_body = D_vector_body
        return total_aerodynamic_forces_body


def get_magnus_effect_forces(lin_vel, ang_vel, TAS,  rho, radius, A_front,
                             alpha, beta):
    """Given the velocity vectors vel and ang_vel (body axes) it provides the
    forces (aerodynamic drag and Magnus effect if it exists) (body axes).
    Data for a soccer ball (smooth sphere).

    Parameters
    ----------
    lin_vel : float array
        [u, v, w]    air velocity body-axes (m/s).
    ang_vel : float array
        [p, q, r]    air velocity body-axes (m/s).
    TAS : float
        true air speed (m/s)
    rho : float
        Air density (ISA) (kg/m3)
    radius : float
        sphere radius (m)
    A_front : float
        Area of a frontal section of the sphere (m)
    alpha : float
        angle of attack (rad).
    beta : float
        sideslip angle (rad).

    Returns
    -------
    F_magnus : float
        magnus force (N).
    F_magnus_vector_body : float array
        magnus Forces vector (1x3) (body axes) (N).

    Notes
    -----
    Smooth ball selected. see [1]

    Sn vs C_magnus table: [1]

    +------------+------------+
    | Sn         | C_magnus   |
    +============+============+
    | 0          |   0        |
    +------------+------------+
    | 0.04       |   0.1      |
    +------------+------------+
    | 0.10       |   0.16     |
    +------------+------------+
    | 0.20       |   0.23     |
    +------------+------------+
    | 0.40       |   0.33     |
    +------------+------------+

    Sn : float
        Sn = wn * radius / V  # Sn is the effective spin number and must take
        values between 0 and 0.4 [1]
    wn : float array
        normal projection of the angular velocity vector over the linear
        velocity vector [1]

    References
    ----------
    .. [1] "Aerodynamics of Sports Balls" Bruce D. Jothmann, January 2007
    .. [2] "Aerodynamics of Sports Balls" Annual Review of Fluid Mechanics,
                                        1875.17:15 Mehta, R.D.
    """
    u, v, w = lin_vel  # components of the linear velocity
    p, q, r = ang_vel  # components of the angular velocity

    # %Obtaining of Magnus force coefficient and Magnus force
    wn = np.sqrt((v * r - w * q) ** 2 + (w * p - u * r) ** 2 +
                 (u * q - v * p) ** 2) / np.sqrt(u ** 2 + v ** 2 + w ** 2)
    # normal projection of the angular velocity vector over the linear velocity
    # vector

    Sn = wn * radius / TAS  # Sn is the effective sping number and must take
    # values between 0 and 0.4 [1]
    check_sn(Sn)

    C_magnus = np.interp(Sn, Sn_list, C_magnus_list)

    check_magnus = np.isclose((v * r - w * q, w * p - u * r, u * q - v * p),
                              (0, 0, 0))
    if check_magnus.all():
        F_magnus = 0
        F_magnus_vector_body = np.array([0, 0, 0])
    else:
        F_magnus = 0.5 * rho * TAS ** 2 * A_front * C_magnus
        dir_F_magnus = - np.array([v * r - w * q, w * p - u * r,
                                  u * q - v * p]) / np.sqrt(
                                  (v * r - w * q) ** 2 + (w * p - u * r) ** 2 +
                                  (u * q - v * p) ** 2)

        F_magnus_vector_body = dir_F_magnus * F_magnus

    return F_magnus_vector_body
