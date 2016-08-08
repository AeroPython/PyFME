"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

OOP Example
-----------

"""
import numpy as np

from pyfme.aircrafts import Cessna310

from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind

from pyfme.models.systems import EulerFlatEarth
from pyfme.simulator import BatchSimulation

aircraft = Cessna310()
atmosphere = ISA1976()
gravity = VerticalConstant()
wind = NoWind()
environment = Environment(atmosphere, gravity, wind)
system = EulerFlatEarth(lat=0, lon=0, h=1000, psi=np.pi/4, x_earth=0, y_earth=0)

not_trimmed_controls = {'delta_elevator': 0.05,
                        'hor_tail_incidence': 0.00,
                        'delta_aileron': 0.00,
                        'delta_rudder': 0.00,
                        'delta_t': 0.5}

controls2trim = ['delta_elevator', 'delta_aileron', 'delta_rudder', 'delta_t']

trimmed_controls, trimmed_system, outputs = aircraft.steady_state_flight_trim(
    system, environment, not_trimmed_controls, TAS=120, gamma=+np.pi/180,
    turn_rate=0.1, controls2trim=controls2trim)

my_simulation = BatchSimulation(aircraft, trimmed_system, environment)

N = 100
time = np.linspace(0, 10, N)
controls = {c_name: np.full((N,), trimmed_controls[c_name]) for
            c_name in trimmed_controls}

my_simulation.set_controls(time, controls)
my_simulation.run_simulation()


