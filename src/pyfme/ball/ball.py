# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:53:17 2016

@author: Juan

"""

import numpy as np
from pyfme.utils.coordinates import wind2body
from pyfme.environment.isa import atm

# Constants
mu = 1.983 * 10 ** -5  # kg/m/s
# Data tables
Re_list = [38000, 100000, 160000, 200000, 250000, 300000, 330000, 350000,
           375000, 400000, 500000, 800000, 200000, 4000000]
Cd_list = [0.49, 0.50, 0.51, 0.51, 0.49, 0.46, 0.39, 0.20, 0.09, 0.07,
           0.07, 0.10, 0.15, 0.18]
Sn_list = [0.00, 0.04, 0.10, 0.20, 0.40]
Cl_list = [0.00, 0.10, 0.16, 0.23, 0.33]


def Geometric_Data(r=0.111):

    """ Provides the value of some geometric data.

    Parameters
    ----
    r    radius(m)

    References
    ----------
    Returns
    ------
    r    radius(m)
    S_circle    Surface (m^2)
    S_sphere    Surface (m^2)
    Vol    Volume (m^3)

    Raises
    ------
    See Also
    --------
    Notes
    -----
    References
    ----------
    .. [1]
    """

    S_circle = np.pi * r ** 2
    S_sphere = 4 * np.pi * r ** 2
    Vol = 4 / 3 * np.pi * r ** 3

    return r, S_circle, S_sphere, Vol


def Mass_and_Inertial_Data(r, mass=0.440):

    """ Provides the value of some mass and inertial data.

    Parameters
    -----
    r    radius (m)
    mass   (kg)

    Returns
    ------
    Ixxb Moment of Inertia x-axis (Kg * m2)
    Iyyb Moment of Inertia y-axis (Kg * m2)
    Izzb Moment of Inertia z-axis (Kg * m2)

    Raises
    ------
    See Also
    --------
    Notes
    -----
    Ball assumed as a sphere shell
    References
    ----------
    .. [1]
    """

    Ixxb = 2 * mass * (r ** 2) / 3
    Iyyb = Ixxb
    Izzb = Ixxb

    I_matrix = np.diag([Ixxb, Iyyb, Izzb])

    return I_matrix


def Ball_aerodynamic_forces(velocity_vector, h, alpha, beta):
    """ Given a velocity vector (body axes) provides the forces (aerodynamic
    drag and Magnus effect) (body axes).
    Data for a soccer ball (smooth sphere).

    Parameters
    -----
    velocity_vector : float array
        [u, v, w, p, q, r]    air velocity body-axes (m/s).
    h : float
        altitude (m).
    alpha : float
        angle of attack (rad).
    beta : float
        sideslip angle (rad).

    Returns
    ------
    Cd : float
        Drag coefficient.
    C_magnus : float
        magnus effect coefficient force.

    D : floeat
        drag force (N).
    D_vector_body : float
        Drag forces vector (body axes) (N).
    F_magnus : float
        magnus force (N).
    F_magnus_vector_body :
        magnus Forces vector (body axes) (N).

    Raises
    ------
    ValueError
        If the value of the Reynolds number is outside the defined layers.
    ValueError
        If the value of the effective spin number is outside the defined
        layers.

    See Also
    --------
    [2]
    Notes
    -----
    Smooth ball selected. see[1]

    Sn          C_magnus    [1]
    --------------------
    0           0
    0.04        0.1
    0.10        0.16
    0.20        0.23
    0.40        0.33

    Re          Cd    [1]
    ----------------
    38000       0.49
    100000      0.50
    160000      0.51
    200000      0.51
    250000      0.49
    300000      0.46
    330000      0.39
    350000      0.20
    375000      0.09
    400000      0.07
    500000      0.07
    800000      0.10
    2000000     0.15
    4000000     0.18

    wn : float array
        normal projection of the angular velocity vector over the linear
        velocity vector
    Sn : float
        Sn = wn * radius / V  # Sn is the effective spin number and must take
        values between 0 and 0.4 [1]
    Re : float
        Re = rho * V * radius / mu  # Reynolds number. values between 38e3 and
        4e6.

    References
    ----------
    "Aerodynamics of Sports Balls" Bruce D. Jothmann, January 2007
    [1]
    "Aerodynamics of Sports Balls" Annual Review of Fluid Mechanics, 1875.17:15
    Mehta, R.D.
    [2]
    """
    u, v, w = velocity_vector[:3]  # components of the linear velocity
    p, q, r = velocity_vector[3:]  # components of the angular velocity
    V = np.linalg.norm(velocity_vector[:3])  # modulus of the linear velocity

    radius, A_front, _, _ = Geometric_Data()
    rho = atm(h)[2]  # density
    Re = rho * V * radius / mu  # Reynolds number

    # %Obtaining of Drag coefficient and Drag force

    if Re < Re_list[0]:
        raise ValueError("Reynolds number cannot be lower than 38000.")
    elif Re > Re_list[-1]:
        raise ValueError("Reynolds number cannot be higher than 4e6.")
    else:
        Cd = np.interp(Re, Re_list, Cd_list)

    D = 0.5 * rho * V ** 2 * A_front * Cd
    D_vector_body = wind2body(([D, 0, 0]), alpha, beta)

    # %Obtaining of Magnus force coefficient and Magnus force

    wn = np.sqrt((v * r - w * q) ** 2 + (w * p - u * r) ** 2 +
                 (u * q - v * p) ** 2) / np.sqrt(u ** 2 + v ** 2 + w ** 2)
    # normal projection of the angular velocity vector over the linear velocity
    # vector

    Sn = wn * radius / V  # Sn is the effective sping number and must take
    # values between 0 and 0.4 [1]

    if Sn < Sn_list[0]:
        raise ValueError("Effective Spin number cannot be less than 0.")
    elif Sn > Sn_list[-1]:
        raise ValueError("Effective Spin number cannot be bigger than 0.40.")
    else:
        C_magnus = np.interp(Sn, Sn_list, Cl_list)

    if v * r - w * q == 0 and w * p - u * r == 0 and u * q - v * p == 0:
        F_magnus = 0
        F_magnus_vector_body = ([0, 0, 0])
    else:
        F_magnus = 0.5 * rho * V ** 2 * A_front * C_magnus
        dir_F_magnus = np.array([v * r - w * q, w * p - u * r,
                                 u * q - v * p]) / wn

        F_magnus_vector_body = dir_F_magnus * F_magnus

    return Cd, D, D_vector_body, C_magnus, F_magnus, F_magnus_vector_body
