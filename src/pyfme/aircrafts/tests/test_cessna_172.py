# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)

from pyfme.aircrafts.cessna_172 import Cessna172


def test_calculate_aero_forces_moments():
    aircraft = Cessna172()
    aircraft.q_inf = 0.5 * 1.225 * 45 ** 2
    aircraft.TAS = 45
    aircraft.alpha = np.deg2rad(5.0)

    aircraft.controls = {'delta_elevator': 0,
                         'hor_tail_incidence': 0,
                         'delta_aileron': 0,
                         'delta_rudder': 0,
                         'delta_t': 0}

    L, D, Y, l, m, n = aircraft._calculate_aero_forces_moments()

    assert_array_almost_equal([L, D, Y],
                              [13060.49063, 964.4670, 0.],
                              decimal=4)

    assert_array_almost_equal([l, m, n],
                              [0, -2046.6386, 0],
                              decimal=2)
