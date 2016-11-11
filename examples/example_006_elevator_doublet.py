# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Example 006
-------

Cessna 172, ISA1976 integrated with Flat Earth (Euler angles).

Evolution of the aircraft after a pitch perturbation (delta doublet
applied to the elevator) at t=1.

Initially trimmed to a stationary, horizontal, symmetric, wings level
flight.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyfme.aircrafts import Cessna172
from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.models.systems import EulerFlatEarth
from pyfme.simulator import BatchSimulation
from pyfme.utils.trimmer import steady_state_flight_trimmer
from pyfme.utils.input_generator import doublet

aircraft = Cessna172()
atmosphere = ISA1976()
gravity = VerticalConstant()
wind = NoWind()
environment = Environment(atmosphere, gravity, wind)

# Initial conditions.
TAS = 45  # m/s
h0 = 2000  # m
psi0 = 1  # rad
x0, y0 = 0, 0  # m
turn_rate = 0.0  # rad/s
gamma0 = 0.00  # rad

system = EulerFlatEarth(lat=0, lon=0, h=h0, psi=psi0, x_earth=x0, y_earth=y0)

not_trimmed_controls = {'delta_elevator': 0.0,
                        'delta_aileron': 0.01 * np.sign(turn_rate),
                        'delta_rudder': 0.01 * np.sign(turn_rate),
                        'delta_t': 0.5}

controls2trim = ['delta_elevator', 'delta_aileron', 'delta_rudder', 'delta_t']

trimmed_ac, trimmed_sys, trimmed_env, results = steady_state_flight_trimmer(
    aircraft, system, environment, TAS=TAS, controls_0=not_trimmed_controls,
    controls2trim=controls2trim, gamma=gamma0, turn_rate=turn_rate, verbose=1)

print()
print('delta_elev = ', "%8.4f" % np.rad2deg(results['delta_elevator']), 'deg')
print('delta_aile = ', "%8.4f" % np.rad2deg(results['delta_aileron']), 'deg')
print('delta_rud = ', "%8.4f" % np.rad2deg(results['delta_rudder']), 'deg')
print('delta_t = ', "%8.4f" % results['delta_t'], '%', '\n')
print('alpha = ', "%8.4f" % np.rad2deg(results['alpha']), 'deg')
print('beta = ', "%8.4f" % np.rad2deg(results['beta']), 'deg', '\n')
print('u = ', "%8.4f" % results['u'], 'm/s')
print('v = ', "%8.4f" % results['v'], 'm/s')
print('w = ', "%8.4f" % results['w'], 'm/s', '\n')
print('psi = ', "%8.4f" % np.rad2deg(psi0), 'deg')
print('theta = ', "%8.4f" % np.rad2deg(results['theta']), 'deg')
print('phi = ', "%8.4f" % np.rad2deg(results['phi']), 'deg', '\n')
print('p =', "%8.4f" % results['p'], 'rad/s')
print('q =', "%8.4f" % results['q'], 'rad/s')
print('r =', "%8.4f" % results['r'], 'rad/s')

my_simulation = BatchSimulation(trimmed_ac, trimmed_sys, trimmed_env)

tfin = 10  # seconds
N = tfin * 100 + 1
time = np.linspace(0, tfin, N)
initial_controls = trimmed_ac.controls

controls = {}
for control_name, control_value in initial_controls.items():
    controls[control_name] = np.ones_like(time) * control_value

# Elevator doublet
# Elevator max travel: +28º/-26º
amplitude = np.deg2rad(20)
controls['delta_elevator'] = doublet(t_init=2,
                                     T=1,
                                     A=amplitude,
                                     time=time,
                                     offset=initial_controls['delta_elevator'])

my_simulation.set_controls(time, controls)

par_list = ['x_earth', 'y_earth', 'height',
            'psi', 'theta', 'phi',
            'u', 'v', 'w',
            'v_north', 'v_east', 'v_down',
            'p', 'q', 'r',
            'alpha', 'beta', 'TAS',
            'F_xb', 'F_yb', 'F_zb',
            'M_xb', 'M_yb', 'M_zb']

my_simulation.set_par_dict(par_list)
my_simulation.run_simulation()

# print(my_simulation.par_dict)

plt.style.use('ggplot')

for ii in range(len(par_list) // 3):
    three_params = par_list[3 * ii:3 * ii + 3]
    fig, ax = plt.subplots(3, 1, sharex=True)
    for jj, par in enumerate(three_params):
        ax[jj].plot(time, my_simulation.par_dict[par])
        ax[jj].set_ylabel(par)
        ax[jj].set_xlabel('time (s)')

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(my_simulation.par_dict['x_earth'],
        my_simulation.par_dict['y_earth'],
        my_simulation.par_dict['height'])

ax.plot(my_simulation.par_dict['x_earth'],
        my_simulation.par_dict['y_earth'],
        my_simulation.par_dict['height'] * 0)
ax.set_xlabel('x_earth')
ax.set_ylabel('y_earth')
ax.set_zlabel('z_earth')

plt.show()
