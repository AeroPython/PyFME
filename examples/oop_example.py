"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

OOP Example
-----------

"""
import numpy as np
import matplotlib.pyplot as plt

from pyfme.aircrafts import Cessna310
from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.models.systems import EulerFlatEarth
from pyfme.simulator import BatchSimulation

aircraft = Cessna310
atmosphere = ISA1976
gravity = VerticalConstant
wind = NoWind
environment = Environment(atmosphere, gravity, wind)
system = EulerFlatEarth(lat=0, lon=0, h=1000, psi=np.pi/4, x_earth=0, y_earth=0)

not_trimmed_controls = {'delta_elevator': 0.05,
                        'hor_tail_incidence': 0.00,
                        'delta_aileron': 0.00,
                        'delta_rudder': 0.00,
                        'delta_t': 0.5}

controls2trim = ['delta_elevator', 'delta_aileron', 'delta_rudder', 'delta_t']

trimmed_controls, trimmed_system, outputs = aircraft.steady_state_flight_trim(
    system=system,
    env=environment,
    controls=not_trimmed_controls,
    TAS=120, gamma=+0*np.pi/180, turn_rate=0.0,
    controls2trim=controls2trim)

my_simulation = BatchSimulation(aircraft, trimmed_system, environment)

tfin = 20  # seconds
N = tfin * 100 + 1
time = np.linspace(0, tfin, N)
controls = {c_name: np.full((N,), trimmed_controls[c_name]) for
            c_name in trimmed_controls}

my_simulation.set_controls(time, controls)

par_list = ['x_earth', 'y_earth', 'height',
            'psi', 'theta', 'phi',
            'u', 'v', 'w',
            'v_north', 'v_east', 'v_down',
            'p', 'q', 'r',
            'alpha', 'beta', 'TAS',
            'T', 'p', 'rho']

my_simulation.set_par_dict(par_list)
my_simulation.run_simulation()

print(my_simulation.par_dict)

plt.style.use('ggplot')

for ii in range(len(par_list) // 3):
    three_params = par_list[3*ii:3*ii+3]
    fig, ax = plt.subplots(3, 1)
    for jj, par in enumerate(three_params):
        ax[jj].plot(time, my_simulation.par_dict[par])
        ax[jj].set_ylabel(par)

plt.show()
