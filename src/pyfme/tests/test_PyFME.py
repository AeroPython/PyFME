import numpy as np

from pyfme.aircrafts import Cessna310
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.environment import Environment
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.models import EulerFlatEarth
from pyfme.models.systems import System
from pyfme.simulator import Simulation

from pyfme.utils.input_generator import Constant, Harmonic


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

    simulation.controls = {'delta_elevator': Constant(0.1),
                           'hor_tail_incidence': Harmonic(1, 3, 1, 2, 0),
                           'delta_aileron': Constant(0.2),
                           'delta_rudder': Constant(0.3),
                           'delta_t': Constant(0.5)}

    simulation.propagate(10)

    for n, v in simulation.results.items():
        print("{}: {}".format(n, v))


def test_trimmer_00():
    aircraft = Cessna310()
    system = System(model=EulerFlatEarth())
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    simulation = Simulation(aircraft, system, environment)

    initial_controls = {'delta_elevator': 0,
                        'hor_tail_incidence': 0,
                        'delta_aileron': 0,
                        'delta_rudder': 0,
                        'delta_t': 0}

    simulation.trim_aircraft(geodetic_initial_pos=(0, 0, 1000),
                             TAS=100,
                             gamma=0.0,
                             turn_rate=0.0,
                             initial_controls=initial_controls,
                             exclude_controls=['hor_tail_incidence'])

    simulation.propagate(10)


def test_trimmer_01():
    aircraft = Cessna310()
    system = System(model=EulerFlatEarth())
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    simulation = Simulation(aircraft, system, environment)

    initial_controls = {'delta_elevator': 0,
                        'hor_tail_incidence': 0,
                        'delta_aileron': 0,
                        'delta_rudder': 0,
                        'delta_t': 0}

    simulation.trim_aircraft(geodetic_initial_pos=(0, 0, 1000),
                             TAS=60,
                             gamma=0.05,
                             turn_rate=0.0,
                             initial_controls=initial_controls,
                             exclude_controls=['hor_tail_incidence'])

    simulation.propagate(10)


def test_trimmer_02():
    aircraft = Cessna310()
    system = System(model=EulerFlatEarth())
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    simulation = Simulation(aircraft, system, environment)

    initial_controls = {'delta_elevator': 0,
                        'hor_tail_incidence': 0,
                        'delta_aileron': 0,
                        'delta_rudder': 0,
                        'delta_t': 0}

    simulation.trim_aircraft(geodetic_initial_pos=(0, 0, 1000),
                             TAS=60,
                             gamma=0.00,
                             turn_rate=0.05,
                             initial_controls=initial_controls,
                             exclude_controls=['hor_tail_incidence'])

    simulation.propagate(10)


def test_trimmer_03():
    aircraft = Cessna310()
    system = System(model=EulerFlatEarth())
    environment = Environment(ISA1976(), VerticalConstant(), NoWind())

    simulation = Simulation(aircraft, system, environment)

    initial_controls = {'delta_elevator': 0,
                        'hor_tail_incidence': 0,
                        'delta_aileron': 0,
                        'delta_rudder': 0,
                        'delta_t': 0}

    simulation.trim_aircraft(geodetic_initial_pos=(0, 0, 1000),
                             TAS=60,
                             gamma=0.05,
                             turn_rate=0.05,
                             initial_controls=initial_controls,
                             exclude_controls=['hor_tail_incidence'])

    simulation.propagate(10)
