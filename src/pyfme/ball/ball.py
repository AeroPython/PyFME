# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:53:17 2016

@author: Juan
"""

import numpy as np
from pyfme.utils.coordinates import wind2body
from pyme.environment.isa import atm


def Geometric_Data(r=0.111):

    """ Provides the value of some geometric data.

    Data
    ----

    r    radius(m)
    S_circle    Surface (m^2)
    S_sphere    Surface (m^2)
    Vol    Volume (m^3)

    References
    ----------
    Return
    ------
    r    radius(m)
    Sw    Surface (m^2)

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

    Data
    -----
    mass   (kg)
    Ixxb Moment of Inertia x-axis (Kg * m2)
    Iyyb Moment of Inertia y-axis (Kg * m2)
    Izzb Moment of Inertia z-axis (Kg * m2)

    Return
    ------
    mass
    I_matrix    Array

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

    Ixxb = 2 * mass * (r ** 2) / 3
    Iyyb = Ixxb
    Izzb = Ixxb

    I_matrix = np.diag([Ixxb, Iyyb, Izzb])

    return mass, I_matrix


def Ball_forces(velocity_vector, rho, alpha, beta):
    """ Given a velocity vector (body axes) provides the forces (body axis).
    Data for a soccer ball (smooth sphere)

    Data
    -----
    A_front    Front section (m^2)
    Cd    Drag coefficient
    C_magnus    magnus effect coefficient force

    D    drag force (Pa)
    Dx_body    drag force x-body-axes(Pa)
    Dy_body    drag force y-body-axes(Pa)
    Dz_body    drag force z-body-axes(Pa)
    D_vector_body    Drag forces vector (body axes) (Pa)

    dir_F_magnus    director vector of the magnus Force (body axes)
    F_magnus    magnus force (Pa)
    F_magnus_vector_body    magnus Forces vector (body axes) (Pa)
    Fx_magnus_body    magnus Forces x-body-axes (Pa)
    Fy_magnus_body    magnus Forces y-body-axes (Pa)
    Fz_magnus_body    magnus Forces z-body-axes (Pa)

    mu = 1.983 e-5   viscosity (kg/s/m)
    radius    (m)
    rho    air density(m/kg^3)
    Re    Reynolds number
    Sn    effective spin number (wn*r/V)
    V    velocity modulus (m/s)
    velocity_vector = np.array([u, v, w, p, q, r])    air velocity body-axes
                                                                          (m/s)
    wn    angular velocity perpendicular to the linear velocity (m/s)

    Sn    C_magnus    [1]
    ---------------
    0     0
    0.04  0.1
    0.10  0.16
    0.20  0.23
    0.40  0.33

    Re       Cd    [1]
    --------------
    38000    0.49
    100000    0.50
    160000    0.51
    200000    0.51
    250000    0.49
    300000    0.46
    330000    0.39
    350000    0.20
    375000    0.09
    400000    0.07
    500000    0.07
    800000    0.10
    2000000    0.15
    4000000    0.18


    Return
    ------

    Raises
    ------
    See Also
    --------
    Notes
    -----
    References
    ----------
    "Aerodynamics of Sports Balls" Annual Review of Fluid Mechanics, 1875.17:15
    [1]
    """

    mu = 1.983 * 10 ** -5
    Re_list = [38000, 100000, 160000, 200000, 250000, 300000, 330000, 350000,
               375000, 400000, 500000, 800000, 200000, 4000000]
    Cd_list = [0.49, 0.50, 0.51, 0.51, 0.49, 0.46, 0.39, 0.20, 0.09, 0.07,
               0.07, 0.10, 0.15, 0.18]
    Sn_list = [0.00, 0.04, 0.10, 0.20, 0.40]
    Cl_list = [0.00, 0.10, 0.16, 0.23, 0.33]

    u, v, w = velocity_vector[:3]
    p, q, r = velocity_vector[3:]
    V = np.linalg.norm(velocity_vector[:3])

    radius, A_front, _, _ = Geometric_Data()
    rho = atm(0)[2]
    Re = rho * V * radius / mu

    if Re < Re_list[0]:
        raise ValueError("Reynolds number cannot be lower than 38000.")
    elif Re > Re_list[-1]:
        raise ValueError("Reynolds number cannot be higher than 4e6.")
    else:
        Cd = np.interp(Re, Re_list, Cd_list)

    wn = np.sqrt((v * r - w * q) ** 2 + (w * p - u * r) ** 2 +
                 (u * q - v * p) ** 2)

    Sn = wn * radius / V

    if Sn < Sn_list[0]:
        raise ValueError("Effective Spin number cannot be less than 0.")
    elif Sn > Sn_list[-1]:
        raise ValueError("Effective Spin number cannot be bigger than 0.40.")
    else:
        C_magnus = np.interp(Sn, Sn_list, Cl_list)

    D = 0.5 * rho * V ** 2 * A_front * Cd

    F_magnus = 0.5 * rho * V ** 2 * A_front * C_magnus
    dir_F_magnus = np.array([v * r - w * q, w * p - u * r, u * q - v * p]) / wn

    F_magnus_vector_body = dir_F_magnus * F_magnus
    Fx_magnus_body = F_magnus_vector_body[0]
    Fy_magnus_body = F_magnus_vector_body[1]
    Fz_magnus_body = F_magnus_vector_body[2]

    D_vector_body = wind2body(([D, 0, 0]), alpha, beta)
    Dx_body = D_vector_body[0]
    Dy_body = D_vector_body[1]
    Dz_body = D_vector_body[2]

    return D, F_magnus, Cd, C_magnus, F_magnus_vector_body, Fx_magnus_body, \
        Fy_magnus_body, Fz_magnus_body, D_vector_body, Dx_body, Dy_body,\
        Dz_body
