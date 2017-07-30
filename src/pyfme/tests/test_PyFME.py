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

    aircraft = Cessna310()
    system = System(model=EulerFlatEarth())
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    simulation = Simulation(aircraft, system, environment)

    simulation.system.set_initial_state(
        geodetic_coordinates=np.array([0., 0., 1000.]),
        vel_body=np.array([100, 0.2, 1]),
        euler_angles=np.array([0.01, 0.0, np.pi])
    )

    controls = {'delta_elevator': 0.1,
                'hor_tail_incidence': 0.2,
                'delta_aileron': 0.1,
                'delta_rudder': 0.3,
                'delta_t': 0.5}

    environment.update(system)

    simulation.propagate(10, controls)
