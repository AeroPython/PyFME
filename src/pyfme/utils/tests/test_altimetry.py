
"""
Tests of altimetry.
"""

import pytest

from pyfme.utils.altimetry import (geometric,
                                   geopotential)


def test1_geometric():

    h = 11000

    z = geometric(h)
    expected_z = 11019.025157232705

    assert z == expected_z


def test2_geometric():

    h = 20000

    z = geometric(h)
    expected_z = 20062.982207526373

    assert z == expected_z


def test1_geopotential():

    z = 0.0

    h = geopotential(z)
    expected_h = 0.0

    assert h == expected_h


def test2_geopotential():

    z = 86000

    h = geopotential(z)
    expected_h = 84854.57642868205

    assert h == expected_h
