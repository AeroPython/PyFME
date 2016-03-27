# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Tests of altimetry
------------------
"""

from numpy.testing import (assert_almost_equal)

from pyfme.utils.altimetry import (geopotential2geometric,
                                   geometric2geopotential)


def test1_geopotential2geometric():

    h = 11000.0

    z = geopotential2geometric(h)
    expected_z = 11019.025157232705

    assert_almost_equal(z, expected_z)


def test2_geopotential2geometric():

    h = 20000.0

    z = geopotential2geometric(h)
    expected_z = 20062.982207526373

    assert_almost_equal(z, expected_z)


def test1_geometric2geopotential():

    z = 0.0

    h = geometric2geopotential(z)
    expected_h = 0.0

    assert_almost_equal(h, expected_h)


def test2_geometric2geopotential():

    z = 86000.0

    h = geometric2geopotential(z)
    expected_h = 84854.57642868205

    assert_almost_equal(h, expected_h)
