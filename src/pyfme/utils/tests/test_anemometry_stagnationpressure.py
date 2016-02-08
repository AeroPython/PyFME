"""
stagnationpressure test function
"""
from numpy.testing import (assert_almost_equal)


from pyfme.utils.anemometry import (stagnation_pressure)
from pyfme.environment.isa import atm

rho_0 = 1.225  # density at sea level (kg/m3)
p_0 = 101325  # pressure at sea level (Pa)
a_0 = 340.293990543  # sound speed at sea level (m/s)
gamma = 1.4  # heat capacity ratio


def test_stagnation_pressure():

    # subsonic case
    _, p, _, a = atm(11000)
    tas = 240

    p_stagnation_expected = 34952.7493849545
    p_stagnation = stagnation_pressure(p, a, tas)
    assert_almost_equal(p_stagnation, p_stagnation_expected)

    # supersonic case
    _, p, _, a = atm(11000)
    tas = 400
    p_stagnation_expected = 65521.299596290904

    p_stagnation = stagnation_pressure(p, a, tas)
    assert_almost_equal(p_stagnation, p_stagnation_expected)
