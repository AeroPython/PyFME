# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 23:56:51 2016

@author:olrosales@gmail.com

@AeroPython
"""

import pytest

import numpy as np

from numpy.testing import (assert_array_almost_equal, assert_almost_equal)

from pyfme.aircrafts.cessna_310 import (get_aerodynamic_forces,
                                        get_aerodynamic_moments,
                                       get_engine_force)


def test_get_aerodynamic_forces():

    TAS = 100
    rho = 1.225
    alpha = 0
    beta = 0
    delta_e = 0
    ih = 0
    delta_ail = 0
    delta_r = 0
    aerodynamic_forces_expected = np.array([-2887.832934, 0, -28679.16845])

    aerodynamic_forces = get_aerodynamic_forces(TAS, rho, alpha,
                                                beta, delta_e, ih, delta_ail,
                                                delta_r)

    assert_array_almost_equal(aerodynamic_forces, aerodynamic_forces_expected,
                              decimal=4)


def test_get_aerodynamic_moments():

    TAS = 100
    rho = 1.225
    alpha = 0
    beta = 0
    delta_e = 0
    ih = 0
    delta_ail = 0
    delta_r = 0
    aerodynamic_moments_expected = np.array([0, 10177.06582, 0])

    aerodynamic_moments = get_aerodynamic_moments(TAS, rho, alpha, beta,
                                                  delta_e, ih, delta_ail,
                                                  delta_r)

    assert_array_almost_equal(aerodynamic_moments,
                              aerodynamic_moments_expected, decimal=4)


def test_get_engine_force():

    delta_t = 0.5

    trust_expected = 1100.399064

    trust = get_engine_force(delta_t)

    assert_almost_equal(trust, trust_expected, decimal=4)

    wrong_value_0 = -0.5
    wrong_value_1 = 2

    with pytest.raises(ValueError):

        get_engine_force(wrong_value_0)
        get_engine_force(wrong_value_1)
