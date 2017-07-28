import numpy as np

from pyfme.models import EulerFlatEarth

mass = 10000
inertia = np.array([[100, 0, 25], [0, 100, 0], [25, 0, 100]])
forces = np.ones(3) * 100
moments = np.ones(3) * 100


def test_euler_flat_earth_dummy_integration_with_callback():

    def foo(t, y):
        print(t, y)
        print(system.time)

    system = EulerFlatEarth(np.zeros(12), callback=foo)
    system.propagate(10, mass, inertia, forces, moments)
