# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Example with trimmed aircraft.
The main purpose of this example is to see the evolution of the aircraft after
a longitudinal perturbation (delta doublet).
Trimmed in stationary, horizontal, symmetric, wings level flight.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyfme.aircrafts import cessna_310
from pyfme.models.system import System
from pyfme.utils.trimmer import steady_state_flight_trim
from pyfme.utils.anemometry import calculate_alpha_beta_TAS
from pyfme.environment.isa import atm


if __name__ == '__main__':

    # Aircraft parameters.
    mass, inertia = cessna_310.mass_and_inertial_data()

    # Initial conditions.
    TAS_ = 312 * 0.3048  # m/s
    h = 8000 * 0.3048  # m
    psi_0 = 3  # rad
    x_0, y_0 = 0, 0  # m

    # Trimming.
    trim_results = steady_state_flight_trim(cessna_310, h, TAS_, gamma=0,
                                            turn_rate=0)

    lin_vel, ang_vel, theta, phi, alpha_, beta_, control_vector = trim_results

    # Time.
    t0 = 0  # s
    tf = 30  # s
    dt = 1e-2  # s

    time = np.arange(t0, tf, dt)

    # Results initialization.
    results = np.zeros([time.size, 12])

    results[0, 0:3] = lin_vel
    results[0, 3:6] = ang_vel
    results[0, 6:9] = theta, phi, psi_0
    results[0, 9:12] = x_0, y_0, h
    alpha = np.empty_like(time)
    alpha[0] = alpha_
    beta = np.empty_like(time)
    beta[0] = beta_
    TAS = np.empty_like(time)
    TAS[0] = TAS_

    # Linear Momentum and Angular Momentum eqs.
    equations = System(integrator='dopri5',
                       model='euler_flat_earth',
                       jac=False)
    u, v, w = lin_vel
    p, q, r = ang_vel

    equations.set_initial_values(u, v, w,
                                 p, q, r,
                                 theta, phi, psi_0,
                                 x_0, y_0, h)

    _, _, rho, _ = atm(h)

    # Define control vectors.
    delta_e, delta_ail, delta_r, delta_t = control_vector
    d_e = np.ones_like(time) * delta_e
    d_e[np.where(time<2)] = delta_e * 1.30
    d_e[np.where(time<1)] = delta_e * 0.70
    d_a = np.ones_like(time) * delta_ail
    d_r = np.ones_like(time) * delta_r
    d_t = np.ones_like(time) * delta_t

    attitude = theta, phi, psi_0

    # Rename function to make it shorter
    forces_and_moments = cessna_310.get_forces_and_moments
    for ii, t in enumerate(time[1:]):

        forces, moments = forces_and_moments(
                            TAS[ii], rho, alpha[ii], beta[ii], d_e[ii], 0,
                            d_a[ii], d_r[ii], d_t[ii], attitude)

        results[ii+1, :] = equations.propagate(mass, inertia, forces, moments,
                                               dt)

        lin_vel = results[ii+1, 0:3]
        ang_vel = results[ii+1, 3:6]
        attitude = results[ii+1, 6:9]
        position = results[ii+1, 9:12]

        alpha[ii+1], beta[ii+1], TAS[ii+1] = calculate_alpha_beta_TAS(*lin_vel)

        _, _, rho, _ = atm(position[2])

    velocities = results[:, 0:6]
    attitude_angles = results[:, 6:9]
    position = results[:, 9:12]

    # PLOTS
    plt.close('all')
    plt.style.use('ggplot')

    plt.figure('pos')
    plt.plot(time, position[:, 0], label='x')
    plt.plot(time, position[:, 1], label='y')
    plt.plot(time, position[:, 2], label='z')
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')
    plt.legend()

    plt.figure('angles')
    plt.plot(time, attitude_angles[:, 0], label='theta')
    plt.plot(time, attitude_angles[:, 1], label='phi')
    plt.plot(time, attitude_angles[:, 2], label='psi')
    plt.xlabel('time (s)')
    plt.ylabel('attitude (rad)')
    plt.legend()

    plt.figure('velocities')
    plt.plot(time, velocities[:, 0], label='u')
    plt.plot(time, velocities[:, 1], label='v')
    plt.plot(time, velocities[:, 2], label='w')
    plt.plot(time, TAS, label='TAS')
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')
    plt.legend()

    plt.figure('ang velocities')
    plt.plot(time, velocities[:, 3], label='p')
    plt.plot(time, velocities[:, 4], label='q')
    plt.plot(time, velocities[:, 5], label='r')
    plt.xlabel('time (s)')
    plt.ylabel('angular velocity (rad/s)')
    plt.legend()

    plt.figure('aero angles')
    plt.plot(time, alpha, label='alpha')
    plt.plot(time, beta, label='beta')
    plt.xlabel('time (s)')
    plt.ylabel('angle (rad)')
    plt.legend()

    plt.figure('2D Trajectory')
    plt.plot(position[:, 0], position[:, 1])
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()

    plt.figure()
    plt.plot(time, alpha, label='alpha')
    plt.plot(time, attitude_angles[:, 0], label='theta')
    plt.plot(time, d_e, label='delta_e')
    plt.plot(time, velocities[:, 4], label='q')
    plt.legend()

    plt.show()
