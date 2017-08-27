# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Example 000
-------

Cessna 172, ISA1976 integrated with Flat Earth (Euler angles).

Evolution of the aircraft with no deflecion of control surfaces.

Initially trimmed.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyfme.aircrafts import Cessna172, Cessna310
from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.models import EulerFlatEarth
from pyfme.models.systems import System
from pyfme.simulator import Simulation

from pyfme.utils.input_generator import Constant, Harmonic, Doublet, Step, Ramp

aircraft = Cessna172()
atmosphere = ISA1976()
gravity = VerticalConstant()
wind = NoWind()
environment = Environment(atmosphere, gravity, wind)

# Initial conditions.
TAS = 45  # m/s
h0 = 2000  # m
psi0 = 1.0  # rad
x0, y0 = 0, 0  # m
turn_rate = 0.0  # rad/s
gamma0 = 0.0  # rad

system = System(EulerFlatEarth())

simulation = Simulation(aircraft, system, environment)

not_trimmed_controls = {'delta_elevator': 0.05,
                        'hor_tail_incidence': 0.0,
                        'delta_aileron': 0.01 * np.sign(turn_rate),
                        'delta_rudder': 0.01 * np.sign(turn_rate),
                        'delta_t': 0.5}
simulation.trim_aircraft((0, 0, h0), TAS, gamma0, turn_rate,
                         not_trimmed_controls, psi0)

trimmed_controls = simulation.aircraft.controls

simulation.controls = {'delta_elevator':
                           Doublet(
                               2, 0.5, 0.1,
                               trimmed_controls['delta_elevator']
                           ),
                       'delta_aileron':
                           Constant(trimmed_controls['delta_aileron']),
                       'delta_rudder':
                           Constant(trimmed_controls['delta_rudder']),
                       'delta_t':
                           Constant(trimmed_controls['delta_t'])
                       }

tfin = 20  # seconds

simulation.propagate(tfin)

kwargs = {'marker':'.'}

simulation.results.plot(y=['x_earth', 'y_earth', 'height'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['psi', 'theta', 'phi'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['v_north', 'v_east', 'v_down'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['p', 'q', 'r'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['alpha', 'beta', 'TAS'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['Fx', 'Fy', 'Fz'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['Mx', 'My', 'Mz'],
                        subplots=True,
                        layout=(3, 1),
                        sharex=True,
                        **kwargs)

simulation.results.plot(y=['elevator', 'aileron', 'rudder', 'thrust'],
                        subplots=True,
                        layout=(4, 1),
                        sharex=True,
                        **kwargs)

print(simulation.results)

plt.show()
