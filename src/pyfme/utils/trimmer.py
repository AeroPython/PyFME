# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Trimmer
-------
This module solves the problem of calculating the values of the state and
control vectors that satisfy the state equations of the aircraft at the initial
condition (or any other given condition. This cannot be done analytically
because of the very complex functional dependence of the erodynamic data.
Instead, it must be done with a numerical algorithm which iteratively adjusts
the independent variables until some solution criterion is met.
"""
from warnings import warn
from copy import deepcopy
import numpy as np
from math import sqrt, sin, cos, tan, atan
from scipy.optimize import least_squares

from pyfme.utils.coordinates import wind2body


def steady_state_flight_trim(ac, system, env, controls, trim_controls_names,
                             verbose=0):
    # TODO: write docstring again
    """Finds a combination of values of the state and control variables that
    correspond to a steady-state flight condition. Steady-state aircraft flight
    can be defined as a condition in which all of the motion variables are
    constant or zero. That is, the linear and angular velocity components are
    constant (or zero), and all acceleration components are zero.

    Parameters
    ----------
    aircraft : aircraft class
        Aircraft class with methods get_forces, get_moments
    h : float
        Geopotential altitude for ISA (m).
    TAS : float
        True Air Speed (m/s).
    gamma : float, optional
        Flight path angle (rad).
    turn_rate : float, optional
        Turn rate, d(psi)/dt (rad/s).

    Returns
    -------
    lin_vel : float array
        [u, v, w] air linear velocity body-axes (m/s).
    ang_vel : float array
        [p, q, r] air angular velocity body-axes (m/s).
    theta : float
        Pitch angle (rad).
    phi : float
        Bank angle (rad).
    alpha : float
        Angle of attack (rad).
    beta : float
        Sideslip angle (rad).
    control_vector : array_like
        [delta_e, delta_ail, delta_r, delta_t].

    Notes
    -----
    See section 3.4 in [1] for the algorithm description.
    See section 2.5 in [1] for the definition of steady-state flight condition.

    References
    ----------
    .. [1] Stevens, BL and Lewis, FL, "Aircraft Control and Simulation",
        Wiley-lnterscience.
    """
    # Copy of system and controls is performed in order to not overwrite the
    # the original attributes.
    trimmmed_system = deepcopy(system)
    trimmed_controls = deepcopy(controls)

    # TODO: try to look for a good inizialization method
    initial_guess = [0.05,  # alpha
                     0.001 * np.sign(system.turn_rate)]  # beta
    for control in trim_controls_names:
        initial_guess.append(controls[control])
    args = (trimmmed_system, ac, env, trim_controls_names, trimmed_controls)

    # TODO: pass max deflection of the controls inside aircraft.
    # lower_bounds = (-1, -0.5, -1, -1, -1, 0)
    # upper_bounds = (+1, +0.5, +1, +1, +1, 1)

    results = least_squares(trimming_cost_func, x0=initial_guess, args=args,
                            verbose=verbose, bounds=(-np.inf, np.inf))

    trimmed_params = results['x']
    fun = results['fun']
    cost = results['cost']

    if cost > 1e-7 or any(abs(fun) > 1e-3):
        warn("Trim process did not converge", RuntimeWarning)
        # TODO: check power settings

    return trimmmed_system, trimmed_controls


def turn_coord_cons(turn_rate, alpha, beta, TAS, gamma=0):
    """Calculates phi for coordinated turn.
    """

    g0 = 9.81
    G = turn_rate * TAS / g0

    if abs(gamma) < 1e-8:
        phi = G * cos(beta) / (cos(alpha) - G * sin(alpha) * sin(beta))
        phi = atan(phi)
    else:
        a = 1 - G * tan(alpha) * sin(beta)
        b = sin(gamma) / cos(beta)
        c = 1 + G**2 * cos(beta)**2

        sq = sqrt(c * (1 - b**2) + G**2 * sin(beta)**2)

        num = (a-b**2) + b * tan(alpha) * sq
        den = a**2 - b**2 * (1 + c * tan(alpha)**2)

        phi = atan(G * cos(beta) / cos(alpha) * num / den)
    return phi


def turn_coord_cons_horizontal_and_small_beta(turn_rate, alpha, TAS):
    """Calculates phi for coordinated turn given that gamma is equal to cero
    and beta is small (beta << 1).
    """

    g0 = 9.81
    G = turn_rate * TAS / g0
    phi = G / cos(alpha)
    phi = atan(phi)
    return phi


def rate_of_climb_cons(gamma, alpha, beta, phi):
    """Calculates theta for the given ROC, wind angles, and roll angle.
    """
    a = cos(alpha) * cos(beta)
    b = sin(phi) * sin(beta) + cos(phi) * sin(alpha) * cos(beta)
    sq = sqrt(a**2 - sin(gamma)**2 + b**2)
    theta = (a * b + sin(gamma) * sq) / (a**2 - sin(gamma)**2)
    theta = atan(theta)
    return theta


def trimming_cost_func(trimmed_params, system, ac, env, trim_controls_names,
                       controls):
    """Function to optimize
    """
    TAS = system.TAS
    gamma = system.gamma

    alpha = trimmed_params[0]
    beta = trimmed_params[1]

    system.alpha = alpha
    system.beta = beta

    for ii, control in enumerate(trim_controls_names):
        controls[control] = trimmed_params[ii+2]

    turn_rate = system.turn_rate
    # Choose coordinated turn constrain equation:
    if abs(turn_rate) < 1e-8:
        phi = 0
    else:
        phi = turn_coord_cons(turn_rate, alpha, beta, TAS, gamma)

    system.euler_angles[2] = phi

    # Rate of climb constrain
    theta = rate_of_climb_cons(gamma, alpha, beta, phi)
    system.euler_angles[1] = theta

    # w = turn_rate * k_h
    # k_h = sin(theta) i_b + sin(phi) * cos(theta) j_b + cos(theta) * sin(phi)
    # w = p * i_b + q * j_b + r * k_b
    p = - turn_rate * sin(theta)
    q = turn_rate * sin(phi) * cos(theta)
    r = turn_rate * cos(theta) * sin(phi)
    system.vel_ang = np.array([p, q, r])

    system.vel_body = wind2body((TAS, 0, 0), alpha, beta)

    forces, moments = ac.get_forces_and_moments(system, controls, env)
    vel = np.concatenate((system.vel_body[:], system.vel_ang[:]))
    output = system.lamceq(0, vel, ac.mass, ac.inertia, forces, moments)
    return output
