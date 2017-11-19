# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Test functions for trimmer
--------------------------

These values are hardcoded from the function results with the current
constants.

"""
from itertools import product

import pytest
from numpy.testing import assert_almost_equal

from pyfme.utils.trimmer import (rate_of_climb_cons, turn_coord_cons,
                                 turn_coord_cons_horizontal_and_small_beta)


def test_rate_of_climb_cons():

    alpha = 0.05
    beta = 0.01
    phi = 0.3
    gamma = 0.2

    expected_theta = 0.25072488304787743

    theta = rate_of_climb_cons(gamma, alpha, beta, phi)

    assert_almost_equal(theta, expected_theta)


def test_turn_coord_cons_horizontal_and_small_beta():

    turn_rate = 0.05
    alpha = 0.05
    TAS = 100

    expected_phi = 0.4720091827041734

    phi = turn_coord_cons_horizontal_and_small_beta(turn_rate, alpha, TAS)

    assert_almost_equal(phi, expected_phi)


def test_turn_coord_cons1_against_2():

    turn_rate = 0.05
    alpha = 0.05
    TAS = 100
    beta = 0
    gamma = 0

    expected_phi = turn_coord_cons_horizontal_and_small_beta(turn_rate,
                                                             alpha,
                                                             TAS)
    phi = turn_coord_cons(turn_rate, alpha, beta, TAS, gamma)

    assert_almost_equal(phi, expected_phi)


def test_turn_coord_cons_small_gamma():

    turn_rate = 0.05
    alpha = 0.05
    TAS = 100
    beta = 0.01
    gamma = 0

    expected_phi = 0.472092273171819

    phi = turn_coord_cons(turn_rate, alpha, beta, TAS, gamma)

    assert_almost_equal(phi, expected_phi)


def test_turn_coord_cons_big_gamma():

    turn_rate = 0.05
    alpha = 0.05
    TAS = 100
    beta = 0.01
    gamma = 0.2

    expected_phi = 0.4767516242692935

    phi = turn_coord_cons(turn_rate, alpha, beta, TAS, gamma)

    assert_almost_equal(phi, expected_phi)


trim_test_data = product(
    [30, 45],
    [500, 1000, 2000],
    [0, 0.005, -0.005],
    [0, 0.01, -0.01, 0.05, -0.05]
)


@pytest.mark.parametrize('TAS, h0, turn_rate, gamma', trim_test_data)
def test_stationary_condition_trimming_Cessna172_ISA1972_NoWind_VerticalConstant(
        TAS, h0, turn_rate, gamma
):
    import numpy as np

    from pyfme.aircrafts import Cessna172
    from pyfme.environment.environment import Environment
    from pyfme.environment.atmosphere import ISA1976
    from pyfme.environment.gravity import VerticalConstant
    from pyfme.environment.wind import NoWind
    from pyfme.models.state import EarthPosition
    from pyfme.utils.trimmer import steady_state_trim

    aircraft = Cessna172()
    atmosphere = ISA1976()
    gravity = VerticalConstant()
    wind = NoWind()
    environment = Environment(atmosphere, gravity, wind)

    # Initial conditions.
    psi0 = 1.0  # rad
    x0, y0 = 0, 0  # m

    pos0 = EarthPosition(x0, y0, h0)

    controls0 = {'delta_elevator': 0.05,
                 'delta_aileron': 0.2 * np.sign(turn_rate),
                 'delta_rudder': 0.2 * np.sign(turn_rate),
                 'delta_t': 0.5,
                 }

    trimmed_state, trimmed_controls = steady_state_trim(
        aircraft, environment, pos0, psi0, TAS, controls0, gamma, turn_rate,
    )

    # Acceleration
    np.testing.assert_almost_equal(
        trimmed_state.acceleration.value,
        np.zeros_like(trimmed_state.acceleration.value),
        decimal=2
    )
    # Angular acceleration
    np.testing.assert_almost_equal(
        trimmed_state.angular_accel.value,
        np.zeros_like(trimmed_state.angular_accel.value),
        decimal=2
    )
