# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

TAS CAS EAS conversion test functions
-------------------------------------
"""
from numpy.testing import (assert_almost_equal)


from pyfme.utils.anemometry import (tas2eas, eas2tas, cas2eas, eas2cas,
                                    tas2cas, cas2tas)
from pyfme.models.constants import RHO_0, P_0, SOUND_VEL_0, GAMMA_AIR
from pyfme.environment.atmosphere import ISA1976


atmosphere = ISA1976()

def test_tas2eas():

    # sea level
    tas = 275
    eas_expected = 275

    eas = tas2eas(tas, RHO_0)

    assert_almost_equal(eas, eas_expected)

    # Test at 11000m
    _, _, rho, _ = atmosphere(11000)
    tas = 275
    eas_expected = 149.88797172756003

    eas = tas2eas(tas, rho)

    assert_almost_equal(eas, eas_expected)


def test_eas2tas():

    # sea level
    eas = 149.88797172756003
    tas_expected = 149.88797172756003

    tas = eas2tas(eas, RHO_0)

    assert_almost_equal(tas, tas_expected)

    # Test at 11000m
    _, _, rho, _ = atmosphere(11000)
    eas = 149.88797172756003
    tas_expected = 275

    tas = eas2tas(eas, rho)

    assert_almost_equal(tas, tas_expected)


def test_tas2cas():

    # sea level
    tas = 275
    cas_expected = 275

    cas = tas2cas(tas, P_0, RHO_0)

    assert_almost_equal(cas, cas_expected)

    # Test at 11000m
    _, p, rho, _ = atmosphere(11000)
    tas = 275
    cas_expected = 162.03569680495048

    cas = tas2cas(tas, p, rho)

    assert_almost_equal(cas, cas_expected)


def test_cas2tas():

    # sea level
    cas = 275
    tas_expected = 275

    tas = cas2tas(cas, P_0, RHO_0)

    assert_almost_equal(tas, tas_expected)

    # Test at 11000m
    _, p, rho, _ = atmosphere(11000)
    cas = 162.03569680495048
    tas_expected = 275

    tas = cas2tas(cas, p, rho)

    assert_almost_equal(tas, tas_expected)


def test_cas2eas():

    # sea level
    cas = 275
    eas_expected = 275

    eas = cas2eas(cas, P_0, RHO_0)

    assert_almost_equal(eas, eas_expected)

    # Test at 11000m
    _, p, rho, _ = atmosphere(11000)
    cas = 162.03569680495048
    eas_expected = 149.88797172756003

    eas = cas2eas(cas, p, rho)

    assert_almost_equal(eas, eas_expected)


def test_eas2cas():

    # sea level
    eas = 275
    cas_expected = 275

    cas = eas2cas(eas, P_0, RHO_0)

    assert_almost_equal(cas, cas_expected)

    # Test at 11000m
    _, p, rho, _ = atmosphere(11000)
    eas = 149.88797172756003
    cas_expected = 162.03569680495048

    cas = eas2cas(eas, p, rho)

    assert_almost_equal(cas, cas_expected)
