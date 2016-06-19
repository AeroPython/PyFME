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

from pyfme.utils.trimmer import steady_state_flight_trim

aircraft = Cessna310()
environment = Environment(ISA1976(), VerticalConstant(), NoWind())
system = EulerFlatEarth()

system.set_initial_flight_conditions(0, 0, 1000, 100, environment)

controls = {'delta_elevator': 0.05,
            'hor_tail_incidence': 0.01,
            'delta_aileron': 0.05,
            'delta_rudder': 0.01,
            'delta_t': 0.5}
controls2trim = ['delta_elevator', 'delta_aileron', 'delta_rudder', 'delta_t']

system, trim_controls = steady_state_flight_trim(aircraft, system, environment,
                        controls, trim_controls_names=controls2trim)
system.set_initial_state_vector()

my_simulation = BatchSimulation(aircraft, system, environment)
N = 100
time = np.linspace(0, 10, N)
controls = {c_name: np.array([trim_controls[c_name]]*N)
            for c_name in trim_controls}
my_simulation.set_aircraft_controls(time, controls)
my_simulation.run_simulation()
