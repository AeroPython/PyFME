"""
TAS CAS EAS convert test functions
"""
import numpy as np
from numpy.testing import (assert_almost_equal)


from pyfme.utils.tascaseas import (tas2eas, eas2tas, cas2eas, eas2cas, tas2cas,
                                   cas2tas)
from pyfme.environment.isa import atm

rho_0 = 1.225  # density at sea level (kg/m3)
p_0 = 101325  # pressure at sea level (Pa)
a_0 = 340.293990543  # sound speed at sea level (m/s)
gamma = 1.4  # heat capacity ratio


def test_tas2eas():

    # sea level
    tas = 275
    eas_expected = 275

    eas = tas2eas(tas, rho_0)

    assert_almost_equal(eas, eas_expected)

    # Test at 11000m
    _, _, rho = atm(11000)
    tas = 275
    eas_expected = 149.88797172756003

    eas = tas2eas(tas, rho)

    assert_almost_equal(eas, eas_expected)


def test_eas2tas():

    # sea level
    eas = 149.88797172756003
    tas_expected = 149.88797172756003

    tas = eas2tas(eas, rho_0)

    assert_almost_equal(tas, tas_expected)

    # Test at 11000m
    _, _, rho = atm(11000)
    eas = 149.88797172756003
    tas_expected = 275

    tas = eas2tas(eas, rho)

    assert_almost_equal(tas, tas_expected)


def test_tas2cas():

    # sea level
    tas = 275
    cas_expected = 275

    cas = tas2cas(tas, p_0, rho_0)

    assert_almost_equal(cas, cas_expected)

    # Test at 11000m
    _, p, rho = atm(11000)
    tas = 275
    cas_expected = 162.03569680495048

    cas = tas2cas(tas, p, rho)

    assert_almost_equal(cas, cas_expected)


def test_cas2tas():

    # sea level
    cas = 275
    tas_expected = 275

    tas = cas2tas(cas, p_0, rho_0)

    assert_almost_equal(tas, tas_expected)

    # Test at 11000m
    _, p, rho = atm(11000)
    cas = 162.03569680495048
    tas_expected = 275

    tas = cas2tas(cas, p, rho)

    assert_almost_equal(tas, tas_expected)


def test_cas2eas():

    # sea level
    cas = 275
    eas_expected = 275

    eas = cas2eas(cas, p_0, rho_0)

    assert_almost_equal(eas, eas_expected)

    # Test at 11000m
    _, p, rho = atm(11000)
    cas = 162.03569680495048
    eas_expected = 149.88797172756003

    eas = cas2eas(cas, p, rho)

    assert_almost_equal(eas, eas_expected)


def test_eas2cas():

    # sea level
    eas = 275
    cas_expected = 275

    cas = eas2cas(eas, p_0, rho_0)

    assert_almost_equal(cas, cas_expected)

    # Test at 11000m
    _, p, rho = atm(11000)
    eas = 149.88797172756003
    cas_expected = 162.03569680495048

    cas = eas2cas(eas, p, rho)

    assert_almost_equal(cas, cas_expected)
