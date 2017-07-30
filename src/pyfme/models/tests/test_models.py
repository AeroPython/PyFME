import numpy as np

from pyfme.models import EulerFlatEarth
from pyfme.models.systems import System

mass = 10000
inertia = np.array([[100, 0, 25], [0, 100, 0], [25, 0, 100]])
forces = np.ones(3) * 100
moments = np.ones(3) * 100


def test_euler_flat_earth_dummy_integration_with_callback():

    def foo(t, y):
        system.model.state = y
        print(t, y)
        print(system.model.time)
        print(system.set_full_system_state(mass, inertia, forces, moments))
        print({name: system.__getattribute__(name) for name in
               system.full_system_state_names})

    model = EulerFlatEarth(np.zeros(12), callback=foo)
    system = System(model=model)
    system.model.propagate(10, mass, inertia, forces, moments)
