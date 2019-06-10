# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Trimmer
-------
This module solves the problem of calculating the values of the state and
control vectors that satisfy the state equations of the aircraft at the
given condition. This cannot be done analytically because of the very complex
functional dependence on the aerodynamic data. Instead, it must be done with
a numerical algorithm which iteratively adjusts the independent variables
until some solution criterion is met.
"""

import copy
from math import sqrt, sin, cos, tan, atan

import numpy as np
from scipy.optimize import least_squares

from pyfme.models import EulerFlatEarth
from pyfme.models.state import AircraftState, EulerAttitude, BodyVelocity
from pyfme.utils.coordinates import wind2body
from pyfme.models.constants import GRAVITY


def steady_state_trim(aircraft, environment, pos, psi, TAS, controls, gamma=0,
                      turn_rate=0, exclude=None, verbose=0):
    """Finds a combination of values of the state and control variables
    that correspond to a steady-state flight condition.

    Steady-state aircraft flight is defined as a condition in which all
    of the motion variables are constant or zero. That is, the linear and
    angular velocity components are constant (or zero), thus all
     acceleration components are zero.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft to be trimmed.
    environment : Environment
        Environment where the aircraft is trimmed including atmosphere,
        gravity and wind.
    pos : Position
        Initial position of the aircraft.
    psi : float, opt
        Initial yaw angle (rad).
    TAS : float
        True Air Speed (m/s).
    controls : dict
        Initial value guess for each control or fixed value if control is
        included in exclude.
    gamma : float, optional
        Flight path angle (rad).
    turn_rate : float, optional
        Turn rate, d(psi)/dt (rad/s).
    exclude : list, optional
        List with controls not to be trimmed. If not given, every control
        is considered in the trim process.
    verbose : {0, 1, 2}, optional
        Level of least_squares verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations (not supported by 'lm'
              method).

    Returns
    -------
    state : AircraftState
        Trimmed aircraft state.
    trimmed_controls : dict
        Trimmed aircraft controls

    Notes
    -----
    See section 3.4 in [1] for the algorithm description.
    See section 2.5 in [1] for the definition of steady-state flight
    condition.

    References
    ----------
    .. [1] Stevens, BL and Lewis, FL, "Aircraft Control and Simulation",
        Wiley-lnterscience.
    """

    # Set initial state
    att0 = EulerAttitude(theta=0, phi=0, psi=psi)
    vel0 = BodyVelocity(u=TAS, v=0, w=0, attitude=att0)
    # Full state
    state0 = AircraftState(pos, att0, vel0)

    # Environment and aircraft are modified in order not to alter their
    # state during trimming process
    environment = copy.deepcopy(environment)
    aircraft = copy.deepcopy(aircraft)

    # Update environment for the current state
    environment.update(state0)

    # Create system: dynamic equations will be used to find the controls and
    # state which generate no u_dot, v_dot, w_dot, p_dot. q_dot, r_dot
    system = EulerFlatEarth(t0=0, full_state=state0)

    # Initialize alpha and beta
    # TODO: improve initialization method
    alpha0 = 0.05
    beta0 = 0.001 * np.sign(turn_rate)

    # For the current alpha, beta, TAS and env, set the aerodynamics of
    # the aircraft (q_inf, CAS, EAS...)
    aircraft._calculate_aerodynamics_2(TAS, alpha0, beta0, environment)

    # Initialize controls
    for control in aircraft.controls:
        if control not in controls:
            raise ValueError(
                "Control {} not given in initial_controls: {}".format(
                    control, controls)
            )
        else:
            aircraft.controls[control] = controls[control]

    if exclude is None:
        exclude = []

    # Select the controls that will be trimmed
    controls_to_trim = list(aircraft.controls.keys() - exclude)

    # Set the variables for the optimization
    initial_guess = [alpha0, beta0]
    for control in controls_to_trim:
        initial_guess.append(controls[control])

    # Set bounds for each variable to be optimized
    lower_bounds = [-0.5, -0.25]  # Alpha and beta lower bounds.
    upper_bounds = [+0.5, +0.25]  # Alpha and beta upper bounds.
    for ii in controls_to_trim:
        lower_bounds.append(aircraft.control_limits[ii][0])
        upper_bounds.append(aircraft.control_limits[ii][1])
    bounds = (lower_bounds, upper_bounds)

    args = (system, aircraft, environment, controls_to_trim, gamma, turn_rate)

    # Trim
    results = least_squares(trimming_cost_func,
                            x0=initial_guess,
                            args=args,
                            verbose=verbose,
                            bounds=bounds)

    # Residuals: last trim_function evaluation
    u_dot, v_dot, w_dot, p_dot, q_dot, r_dot = results.fun

    att = system.full_state.attitude
    system.full_state.acceleration.update([u_dot, v_dot, w_dot], att)
    system.full_state.angular_accel.update([p_dot, q_dot, r_dot], att)

    trimmed_controls = controls
    for key, val in zip(controls_to_trim, results.x[2:]):
        trimmed_controls[key] = val

    return system.full_state, trimmed_controls


def turn_coord_cons(turn_rate, alpha, beta, TAS, gamma=0):
    """Calculates phi for coordinated turn.
    """

    g0 = GRAVITY
    G = turn_rate * TAS / g0

    if abs(gamma) < 1e-8:
        phi = G * cos(beta) / (cos(alpha) - G * sin(alpha) * sin(beta))
        phi = atan(phi)
    else:
        a = 1 - G * tan(alpha) * sin(beta)
        b = sin(gamma) / cos(beta)
        c = 1 + G ** 2 * cos(beta) ** 2

        sq = sqrt(c * (1 - b ** 2) + G ** 2 * sin(beta) ** 2)

        num = (a - b ** 2) + b * tan(alpha) * sq
        den = a ** 2 - b ** 2 * (1 + c * tan(alpha) ** 2)

        phi = atan(G * cos(beta) / cos(alpha) * num / den)
    return phi


def turn_coord_cons_horizontal_and_small_beta(turn_rate, alpha, TAS):
    """Calculates phi for coordinated turn given that gamma is equal to zero
    and beta is small (beta << 1).
    """

    g0 = GRAVITY
    G = turn_rate * TAS / g0
    phi = G / cos(alpha)
    phi = atan(phi)
    return phi


def rate_of_climb_cons(gamma, alpha, beta, phi):
    """Calculates theta for the given ROC, wind angles, and roll angle.
    """
    a = cos(alpha) * cos(beta)
    b = sin(phi) * sin(beta) + cos(phi) * sin(alpha) * cos(beta)
    sq = sqrt(a ** 2 - sin(gamma) ** 2 + b ** 2)
    theta = (a * b + sin(gamma) * sq) / (a ** 2 - sin(gamma) ** 2)
    theta = atan(theta)
    return theta


def trimming_cost_func(trimmed_params, system, aircraft, environment,
                       controls2trim, gamma, turn_rate):
    """Function to optimize
    """
    alpha = trimmed_params[0]
    beta = trimmed_params[1]

    new_controls = {}
    for ii, control in enumerate(controls2trim):
        new_controls[control] = trimmed_params[ii + 2]

    # Choose coordinated turn constrain equation:
    if abs(turn_rate) < 1e-8:
        phi = 0
    else:
        phi = turn_coord_cons(turn_rate, alpha, beta, aircraft.TAS, gamma)

    # Rate of climb constrain
    theta = rate_of_climb_cons(gamma, alpha, beta, phi)

    # w = turn_rate * k_h
    # k_h = sin(theta) i_b + sin(phi) * cos(theta) j_b + cos(theta) * sin(phi)
    # w = p * i_b + q * j_b + r * k_b
    p = - turn_rate * sin(theta)
    q = turn_rate * sin(phi) * cos(theta)
    r = turn_rate * cos(theta) * cos(phi)

    u, v, w = wind2body((aircraft.TAS, 0, 0), alpha=alpha, beta=beta)

    psi = system.full_state.attitude.psi
    system.full_state.attitude.update([theta, phi, psi])
    attitude = system.full_state.attitude

    system.full_state.velocity.update([u, v, w], attitude)
    system.full_state.angular_vel.update([p, q, r], attitude)
    system.full_state.acceleration.update([0, 0, 0], attitude)
    system.full_state.angular_accel.update([0, 0, 0], attitude)

    rv = system.steady_state_trim_fun(system.full_state, environment, aircraft,
                                      new_controls)
    return rv
