# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

<Module name>
<Brief description ...>
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Dummy example

@author: asaez
"""

import numpy as np
import matplotlib.pyplot as plt

from pyfme.aircrafts import cessna_310
from pyfme.models.system import System
from pyfme.utils.trimmer import steady_state_flight_trim
from pyfme.environment.isa import atm


if __name__ == '__main__':

    # Case params
    mass, inertia = cessna_310.mass_and_inertial_data()

    # initial conditions
    TAS = 85  # m/s
    h = 3000  # m
    psi_0 = 3  # rad
    x_0, y_0 = 0, 0  # m

    trim_results = steady_state_flight_trim(cessna_310, h, TAS, gamma=0.5,
                                            turn_rate=0)

    lin_vel, ang_vel, theta, phi, alpha, beta, control_vector = trim_results
    # Time
    t0 = 0
    tf = 30
    dt = 1e-1
    time = np.arange(t0, tf, dt)

    # Results initialization
    results = np.zeros([time.size, 12])

    velocities = np.empty([time.size, 6])
    results[0, 0:3] = lin_vel
    results[0, 3:6] = ang_vel
    results[0, 6:9] = theta, phi, psi_0
    results[0, 9:12] = x_0, y_0, h

    # Linear Momentum and Angular Momentum eqs
    equations = System(integrator='dopri5', model='euler_flat_earth', jac=False)
    u, v, w = lin_vel
    p, q, r = ang_vel
    equations.set_initial_values(u, v, w, p, q, r, theta, phi, psi_0, x_0, y_0,
                                 h)

    _, _, rho, _ = atm(h)

    delta_e, delta_ail, delta_r, delta_t = control_vector
    attitude = theta, phi, psi_0

    for ii, t in enumerate(time[1:]):
        forces, moments = cessna_310.get_forces_and_moments(TAS, rho, alpha, beta,
                                                    delta_e, 0, delta_ail, delta_r, delta_t,
                                                    attitude)

        results[ii+1, :] = equations.propagate(mass, inertia, forces, moments,
                                               dt)

        lin_vel = results[ii+1, 0:3]
        ang_vel = results[ii+1, 3:6]
        attitude = results[ii+1, 6:9]
        position = results[ii+1, 9:12]

        _, _, rho, _ = atm(position[2])

        TAS = np.sqrt(np.sum(lin_vel*lin_vel))


velocities = results[:, 0:6]
attitude_angles = results[:, 6:9]
position = results[:, 9:12]

plt.close('all')
plt.style.use('ggplot')

# TODO: improve graphics: legend title....
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
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.legend()

plt.figure('ang velocities')
plt.plot(time, velocities[:, 3], label='p')
plt.plot(time, velocities[:, 4], label='q')
plt.plot(time, velocities[:, 5], label='r')
plt.xlabel('time (s)')
plt.xlabel('angular velocity (rad/s)')
plt.legend()