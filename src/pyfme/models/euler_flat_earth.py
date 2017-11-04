# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Euler Flat Earth
----------------

Classical aircraft motion equations assuming no Earth rotation
inertial effects, representing attitude with Euler angles (not valid for
all-attitude flight) and integrating aircraft position in Earth axis (Flat
Earth).
"""

import numpy as np
from numpy import sin, cos

from pyfme.models.dynamic_system import DynamicSystem


class EulerFlatEarth(DynamicSystem):

    n_states = 12

    def __init__(self, t0, x0, update, method='Rk45', options=None):

        # TODO: use jacobian when it is calculated and implemented in
        # Dynamic System
        super().__init__(t0, x0, method=method, options=options)

        self.update_simulation = update

    def fun(self, t, x):

        updated_simulation = self.update_simulation(t, x)

        mass = updated_simulation.aircraft.mass
        inertia = updated_simulation.aircraft.inertia
        forces = updated_simulation.aircraft.forces
        moments = updated_simulation.aircraft.moments

        rv = _system_equations(t, x, mass, inertia, forces, moments)

        return rv


# TODO: numba jit
def _system_equations(time, state_vector, mass, inertia, forces, moments):
    """Euler flat earth equations: linear momentum equations, angular momentum
    equations, angular kinematic equations, linear kinematic
    equations.

    Parameters
    ----------
    time : float
        Current time (s).
    state_vector : array_like, shape(9)
        Current value of absolute velocity and angular velocity, both
        expressed in body axes, euler angles and position in Earth axis.
        (u, v, w, p, q, r, theta, phi, psi, x, y, z)
         (m/s, m/s, m/s, rad/s, rad/s rad/s, rad, rad, rad, m, m ,m).
    mass : float
        Current mass of the aircraft (kg).
    inertia : array_like, shape(3, 3)
        3x3 tensor of inertia of the aircraft (kg * m2)
        Current equations assume that the aircraft has a symmetry plane
        (x_b - z_b), thus J_xy and J_yz must be null.
    forces : array_like, shape(3)
        3 dimensional vector containing the total total_forces (including
        gravity) in x_b, y_b, z_b axes (N).
    moments : array_like, shape(3)
        3 dimensional vector containing the total total_moments in x_b,
        y_b, z_b axes (N·m).

    Returns
    -------
    dstate_dt : array_like, shape(9)
        Derivative with respect to time of the state vector.
        Current value of absolute acceleration and angular acceleration,
        both expressed in body axes, Euler angles derivatives and velocity
        with respect to Earth Axis.
        (du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dtheta_dt, dphi_dt,
        dpsi_dt, dx_dt, dy_dt, dz_dt)
        (m/s² , m/s², m/s², rad/s², rad/s², rad/s², rad/s, rad/s, rad/s,
        m/s, m/s, m/s).

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight", Courier Corporation,
        p. 149 (5.8 The Flat-Earth Approximation), 2012.

    .. [2] M. A. Gómez Tierno y M. Pérez Cortés, "Mecánica del Vuelo",
        Garceta Grupo Editorial, pp.18-25 (Tema 2: Ecuaciones Generales del
        Moviemiento), 2012.

    """
    # Note definition of total_moments of inertia p.21 Gomez Tierno, et al
    # Mecánica de vuelo
    Ix = inertia[0, 0]
    Iy = inertia[1, 1]
    Iz = inertia[2, 2]
    Jxz = - inertia[0, 2]

    Fx, Fy, Fz = forces
    L, M, N = moments

    u, v, w = state_vector[0:3]
    p, q, r = state_vector[3:6]
    theta, phi, psi = state_vector[6:9]

    # Linear momentum equations
    du_dt = Fx / mass + r * v - q * w
    dv_dt = Fy / mass - r * u + p * w
    dw_dt = Fz / mass + q * u - p * v

    # Angular momentum equations
    dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
             p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)
    dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
    dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
             q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)

    # Angular Kinematic equations
    dtheta_dt = q * cos(phi) - r * sin(phi)
    dphi_dt = p + (q * sin(phi) + r * cos(phi)) * np.tan(theta)
    dpsi_dt = (q * sin(phi) + r * cos(phi)) / cos(theta)

    # Linear kinematic equations
    dx_dt = (cos(theta) * cos(psi) * u +
             (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) * v +
             (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) * w)
    dy_dt = (cos(theta) * sin(psi) * u +
             (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) * v +
             (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) * w)
    dz_dt = -u * sin(theta) + v * sin(phi) * cos(theta) + w * cos(
        phi) * cos(theta)

    return np.array([du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dtheta_dt,
                     dphi_dt, dpsi_dt, dx_dt, dy_dt, dz_dt])
