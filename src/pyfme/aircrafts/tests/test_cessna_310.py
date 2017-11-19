# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Created on Sat Jan  9 23:56:51 2016

@author:olrosales@gmail.com

@AeroPython
"""

from numpy.testing import assert_array_almost_equal

from pyfme.aircrafts.cessna_310 import Cessna310


def test_calculate_aero_forces_moments_alpha_beta_zero():
    aircraft = Cessna310()
    aircraft.q_inf = 0.5 * 1.225 * 100 ** 2
    aircraft.alpha = 0
    aircraft.beta = 0

    aircraft.controls = {'delta_elevator': 0,
                         'hor_tail_incidence': 0,
                         'delta_aileron': 0,
                         'delta_rudder': 0,
                         'delta_t': 0}

    L, D, Y, l, m, n = aircraft._calculate_aero_forces_moments()
    assert_array_almost_equal([L, D, Y],
                              [28679.16845, 2887.832934, 0.],
                              decimal=4)
    assert_array_almost_equal([l, m, n],
                              [0, 10177.065816, 0],
                              decimal=4)


