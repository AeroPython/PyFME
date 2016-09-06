# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Test functions for trimmer

These values are hardcoded from the function results with the current costants.

--------------------------
"""

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
