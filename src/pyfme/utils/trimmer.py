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
import numpy as np
from math import sqrt, sin, cos, tan, atan

from pyfme.utils.coordinates import wind2body


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
                       gamma, turn_rate):
    """Function to optimize
    """
    ac.alpha = trimmed_params[0]
    ac.beta = trimmed_params[1]

    for ii, control in enumerate(trim_controls_names):
        ac.controls[control] = trimmed_params[ii+2]

    # Choose coordinated turn constrain equation:
    if abs(turn_rate) < 1e-8:
        phi = 0
    else:
        phi = turn_coord_cons(turn_rate, ac.alpha, ac.beta, ac.TAS, gamma)

    system.euler_angles[2] = phi

    # Rate of climb constrain
    theta = rate_of_climb_cons(gamma, ac.alpha, ac.beta, phi)
    system.euler_angles[1] = theta

    # w = turn_rate * k_h
    # k_h = sin(theta) i_b + sin(phi) * cos(theta) j_b + cos(theta) * sin(phi)
    # w = p * i_b + q * j_b + r * k_b
    p = - turn_rate * sin(theta)
    q = turn_rate * sin(phi) * cos(theta)
    r = turn_rate * cos(theta) * sin(phi)
    system.vel_ang = np.array([p, q, r])

    system.vel_body = wind2body((ac.TAS, 0, 0), ac.alpha, ac.beta)

    env.update(system)
    ac.gravity_force = env.gravity_vector * ac.mass

    forces, moments = ac.calculate_forces_and_moments()
    vel = np.concatenate((system.vel_body[:], system.vel_ang[:]))
    output = system.lamceq(0, vel, ac.mass, ac.inertia, forces, moments)
    return output
