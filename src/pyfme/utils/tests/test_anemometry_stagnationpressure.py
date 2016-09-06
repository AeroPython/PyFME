# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Stagnation pressure test function
---------------------------------
"""
from numpy.testing import (assert_almost_equal)

from pyfme.utils.anemometry import stagnation_pressure
from pyfme.environment.atmosphere import ISA1976

atmosphere=ISA1976()

def test_stagnation_pressure():

    # subsonic case
    _, p, _, a = atmosphere(11000)
    tas = 240

    #This value is hardcoded from the function result with the current costants

    p_stagnation_expected = 34962.95478339375 
    p_stagnation = stagnation_pressure(p, a, tas)
    assert_almost_equal(p_stagnation, p_stagnation_expected)

    #This value is hardcoded from the function result with the current costants

    # supersonic case
    _, p, _, a = atmosphere(11000)
    tas = 400
    p_stagnation_expected = 65558.23180026768

    p_stagnation = stagnation_pressure(p, a, tas)
    assert_almost_equal(p_stagnation, p_stagnation_expected)
