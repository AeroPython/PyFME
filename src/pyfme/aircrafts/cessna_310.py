# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Cessna 310
----------
Created on Sun Jan  3 18:44:39 2016

@author:olrosales@gmail.com

@AeroPython
"""

# Aircraft = Cessna 310


import numpy as np

from pyfme.utils.anemometry import calculate_dynamic_pressure
from pyfme.utils.coordinates import hor2body


def geometric_data():

    """ Provides the value of some geometric data.

    Returns
    ----
    Sw : float
         Wing surface (m2).
    c : foat
        Mean aerodynamic Chord (m).
    span : float
         Wing span (m).

    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """

    Sw = 175 * 0.3048 * 0.3048    # m2
    c = 4.79 * 0.3048   # m
    span = 36.9 * 0.3048   # m

    return Sw, c, span


def mass_and_inertial_data():

    """ Provides the value of some mass and inertial data.

    Returns
    ------
    mass : float
        mass (lb * 0.453592 = kg)
    inertia : array_like
        Inertia tensor (Kg * m^2)

    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """

    mass = 4600 * 0.453592   # kg
    Ixx_b = 8884 * 1.3558179   # Kg * m2
    Iyy_b = 1939 * 1.3558179   # Kg * m2
    Izz_b = 11001 * 1.3558179   # Kg * m2
    Ixz_b = 0 * 1.3558179   # Kg * m2

    inertia = np.diag([Ixx_b, Iyy_b, Izz_b])

    return mass, inertia


def long_aero_coefficients():

    """Assigns the value of the coefficients
    of stability in cruise conditions and order them in a matrix.


    CL_0 is the lift coefficient evaluated at the initial condition
    CL_a is the lift stability derivative with respect to the angle of attack
    CL_de is the lift stability derivative with respect to the elevator
         deflection
    CL_dih is the lift stability derivative with respect to the stabilator
         deflection

    CD_0 is the drag coefficient evaluated at the initial condition
    CD_a is the drag stability derivative with respect to the angle of attack

    Cm_0 is the pitching moment coefficient evaluated at the condition
        (alpha0 = deltaE = deltaih = 0º)
    Cm_a is the pitching moment stability derivative with respect to the angle
        of attack
    Cm_de is the pitching moment stability derivative with respect to the
        elevator deflection
    Cm_dih is the pitching moment stability derivative with respect to the
         stabilator deflection

    Returns
    -------
    Long_coef_matrix : array_like
                                [
                                [CL_0, CL_a, CL_de, CL_dih],
                                [CD_0, CD_a, 0, 0],
                                [Cm_0, Cm_a, Cm_de, Cm_dih]
                                ]


    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """
    CL_0 = 0.288
    CL_a = 4.58
    CL_de = 0.81
    CL_dih = 0

    CD_0 = 0.029
    CD_a = 0.160

    Cm_0 = 0.07
    Cm_a = -0.137
    Cm_de = -2.26
    Cm_dih = 0

    long_coef_matrix = np.array([
                                [CL_0, CL_a, CL_de, CL_dih],
                                [CD_0, CD_a, 0, 0],
                                [Cm_0, Cm_a, Cm_de, Cm_dih]
                                ])

    return long_coef_matrix


def lat_aero_coefficients():

    """Assigns the value of the coefficients
    of stability in cruise conditions and order them in a matrix.

    CY_b is the side force stability derivative with respect to the
        angle of sideslip
    CY_da is the side force stability derivative with respect to the
         aileron deflection
    CY_dr is the side force stability derivative with respect to the
         rudder deflection

    Cl_b is the rolling moment stability derivative with respect to
        angle of sideslip
    Cl_da is the rolling moment stability derivative with respect to
        the aileron deflection
    Cl_dr is the rolling moment stability derivative with respect to
        the rudder deflection

    Cn_b is the yawing moment stability derivative with respect to the
        angle of sideslip
    Cn_da is the yawing moment stability derivative with respect to the
        aileron deflection
    Cn_dr is the yawing moment stability derivative with respect to the
        rudder deflection

    returns
    -------
    Long_coef_matrix : array_like
                                [
                                [CY_b, CY_da, CY_dr],
                                [Cl_b, Cl_da, Cl_dr],
                                [Cn_b, Cn_da, Cn_dr]
                                ]


    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 590
    """

    CY_b = -0.698
    CY_da = 0
    CY_dr = 0.230

    Cl_b = -0.1096
    Cl_da = 0.172
    Cl_dr = 0.0192

    Cn_b = 0.1444
    Cn_da = -0.0168
    Cn_dr = -0.1152

    lat_coef_matrix = np.array([
                                [CY_b, CY_da, CY_dr],
                                [Cl_b, Cl_da, Cl_dr],
                                [Cn_b, Cn_da, Cn_dr]
                                ])

    return lat_coef_matrix


def get_aerodynamic_forces(TAS, rho, alpha, beta, delta_e, ih, delta_ail,
                           delta_r):

    """ Calculates forces

    Parameters
    ----------

    rho : float
        density (kg/(m3).
    TAS : float
        velocity (m/s).
    alpha : float
        attack angle (rad).
    beta : float
        sideslip angle (rad).
    delta_e : float
        elevator deflection (rad).
    ih : float
        stabilator deflection (rad).
    delta_ail : float
        aileron deflection (rad).
    delta_r : float
        rudder deflection (rad).

    Returns
    -------
    forces : array_like
        3 dimensional vector with (F_x_s, F_y_s, F_z_s) forces  in stability
        axes.

    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    chapter 3 and 4
    """

    long_coef_matrix = long_aero_coefficients()
    lat_coef_matrix = lat_aero_coefficients()

    CL_0, CL_a, CL_de, CL_dih = long_coef_matrix[0, :]
    CD_0, CD_a, CD_de, CD_dih = long_coef_matrix[1, :]
    CY_b, CY_da, CY_dr = lat_coef_matrix[0, :]

    CL_full = CL_0 + CL_a * alpha + CL_de * delta_e + CL_dih * ih
    CD_full = CD_0 + CD_a * alpha + CD_de * delta_e + CD_dih * ih
    CY_full = CY_b * beta + CY_da * delta_ail + CY_dr * delta_r

    Sw = geometric_data()[0]

    aerodynamic_forces = calculate_dynamic_pressure(rho, TAS) *\
        Sw * np.array([-CD_full, CY_full, -CL_full])   # N

    return aerodynamic_forces


def get_aerodynamic_moments(TAS, rho, alpha, beta, delta_e, ih, delta_ail,
                            delta_r):

    """ Calculates forces

    Parameters
    ----------
    rho : float
        density (kg/m3).
    TAS : float
        velocity (m/s).
    alpha : float
        attack angle (rad).
    beta : float
        sideslip angle (rad).
    delta_e : float
        elevator deflection (rad).
    ih : float
        stabilator deflection (rad).
    delta_ail : float
        aileron deflection (rad).
    delta_r : float
        rudder deflection (rad).

    Returns
    -------
    moments : array_like
         3 dimensional vector with (Mxs, Mys, Mzs) forces
         in stability axes.

    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    chapter 3 and 4
    """

    long_coef_matrix = long_aero_coefficients()
    lat_coef_matrix = lat_aero_coefficients()

    Cm_0, Cm_a, Cm_de, Cm_dih = long_coef_matrix[2, :]
    Cl_b, Cl_da, Cl_dr = lat_coef_matrix[1, :]
    Cn_b, Cn_da, Cn_dr = lat_coef_matrix[2, :]

    Cm_full = Cm_0 + Cm_a * alpha + Cm_de * delta_e + Cm_dih * ih
    Cl_full = Cl_b * beta + Cl_da * delta_ail + Cl_dr * delta_r
    Cn_full = Cn_b * beta + Cn_da * delta_ail + Cn_dr * delta_r

    span = geometric_data()[2]
    c = geometric_data()[1]
    Sw = geometric_data()[0]

    aerodynamic_moments = calculate_dynamic_pressure(rho, TAS) * Sw\
        * np.array([Cl_full * span, Cm_full * c, Cn_full * span])

    return aerodynamic_moments


def get_engine_force(delta_t):
    """ Calculates forces

    Parameters
    ----------
    delta_t : float
        trust_lever (0 = 0 Newton, 1 = CT).

    returns
    -------
    engine_force : float
        Thrust (N).

    References
    ----------
    Airplane Flight Dyanamics and Automatic Flight Controls part I - Jan Roskam
    """
    if 0 <= delta_t <= 1:

        Ct = 0.031 * delta_t

        q_cruise = 91.2 * 47.880172   # Pa

        Sw = geometric_data()[0]

        engine_force = Ct * Sw * q_cruise   # N

    else:
        raise ValueError('delta_t must be between 0 and 1')

    return engine_force


def get_forces_and_moments(TAS, rho, alpha, beta, delta_e, ih, delta_ail,
                           delta_r, delta_t, attitude):
    """Return the total forces and moments including aerodynamics, thrust and
    gravity.

    Parameters
    ----------
    TAS : float
        velocity (m/s).
    rho : float
        density (kg/m3).
    alpha : float
        attack angle (rad).
    beta : float
        sideslip angle (rad).
    delta_e : float
        elevator deflection (rad).
    ih : float
        stabilator deflection (rad).
    delta_ail : float
        aileron deflection (rad).
    delta_r : float
        rudder deflection (rad).
    delta_t : float
        Thrust level (between 0-1).
    attitude : array_like
        Attitude angles: (theta, phi, psi).

    Returns
    -------
    forces : array_like
        3 dimensional vector with (F_x_s, F_y_s, F_z_s) forces  in body axes.
        (N)
    moments : array_like
         3 dimensional vector with (Mxs, Mys, Mzs) forces in body axes. (N·m)
    """
    # As stability axes are coincident with wind axes at the moment of
    # linearization (with alpha=0 and beta=0), stability axes are parallel to
    # body axes.
    aero_forces = get_aerodynamic_forces(TAS, rho, alpha, beta, delta_e, 0,
                                         delta_ail, delta_r)
    thrust = get_engine_force(delta_t)
    # Assuming that all the engine thrust is prodecued in the x_body direction
    engine_force = np.array([thrust, 0, 0])

    g0 = 9.81  # m/s²
    theta, phi, psi = attitude
    gravity_force = hor2body((0, 0, g0), theta, phi, psi)

    forces = aero_forces + engine_force + gravity_force

    # It is assumed that the engine moments are zero.
    moments = get_aerodynamic_moments(TAS, rho, alpha, beta, delta_e, ih,
                                      delta_ail, delta_r)

    return forces, moments
