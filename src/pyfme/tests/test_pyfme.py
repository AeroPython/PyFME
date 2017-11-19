from pyfme.aircrafts import Cessna172
from pyfme.environment.environment import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind
from pyfme.models.state.position import EarthPosition
from pyfme.utils.trimmer import steady_state_trim
from pyfme.models import EulerFlatEarth
from pyfme.simulator import Simulation

from pyfme.utils.input_generator import Constant, Doublet


def test_simulation():

    atmosphere = ISA1976()
    gravity = VerticalConstant()
    wind = NoWind()

    environment = Environment(atmosphere, gravity, wind)
    aircraft = Cessna172()

    initial_position = EarthPosition(0, 0, 1000)

    controls_0 = {'delta_elevator': 0.05,
                  'delta_aileron': 0,
                  'delta_rudder': 0,
                  'delta_t': 0.5,
                  }

    trimmed_state, trimmed_controls = steady_state_trim(
        aircraft, environment, initial_position, psi=1, TAS=50,
        controls=controls_0
    )

    system = EulerFlatEarth(t0=0, full_state=trimmed_state)

    controls = {
        'delta_elevator': Doublet(2, 1, 0.1,
                                  trimmed_controls['delta_elevator']),
        'delta_aileron': Constant(trimmed_controls['delta_aileron']),
        'delta_rudder': Constant(trimmed_controls['delta_rudder']),
        'delta_t': Constant(trimmed_controls['delta_t'])
    }

    simulation = Simulation(aircraft, system, environment, controls)
    simulation.propagate(10)
