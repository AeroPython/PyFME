# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Anemometry test functions
-------------------------
"""

from math import sqrt, atan, asin
from numpy.testing import assert_almost_equal

from pyfme.utils.anemometry import (calculate_alpha_beta_TAS,
                                    calculate_dynamic_pressure)


def test_calculate_alpha_beta_TAS():

    u, v, w = 10, 0, 10

    expected_TAS = sqrt(200)
    expected_alfa = atan(1)
    expected_beta = 0

    alfa, beta, TAS = calculate_alpha_beta_TAS(u, v, w)

    assert_almost_equal((expected_alfa, expected_beta, expected_TAS),
                        (alfa, beta, TAS))

    u, v, w = 10, 10, 0

    expected_alfa = 0
    expected_beta = asin(10 / sqrt(200))

    alfa, beta, TAS = calculate_alpha_beta_TAS(u, v, w)

    assert_almost_equal((expected_alfa, expected_beta, expected_TAS),
                        (alfa, beta, TAS))


def test_calculate_dynamic_pressure():

    rho = 1
    TAS = 1

    expected_q_inf = 0.5

    q_inf = calculate_dynamic_pressure(rho, TAS)

    assert_almost_equal(expected_q_inf, q_inf)
