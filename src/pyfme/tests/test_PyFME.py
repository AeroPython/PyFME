import numpy as np

from pyfme.aircrafts import Cessna310
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.environment import Environment
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.models import EulerFlatEarth
from pyfme.models.systems import System
from pyfme.simulator import Simulation


def test_simulation():

    def foo(t, y):
        print(t, y)

    aircraft = Cessna310()
    # XXX: this initial condition does not set the full system state
    x0 = np.array([100, 0, 1, 0, 0, 0, 0.05, 0.02, 0, 0, 0, 1000])
    system = System(model=EulerFlatEarth(callback=foo))
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    simulation = Simulation(aircraft, system, environment)

    controls = {'delta_elevator': 0.1,
                'hor_tail_incidence': 0.2,
                'delta_aileron': 0.1,
                'delta_rudder': 0.3,
                'delta_t': 0.5}

    environment.update(system)

    simulation.set_initial_state(x0, controls)

    simulation.propagate(10, controls)
