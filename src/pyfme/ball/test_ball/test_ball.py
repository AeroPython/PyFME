# -*- coding: utf-8 -*-
"""
Frames of Reference orientation test functions
"""
import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)


from pyfme.ball.ball import (Geometric_Data, Mass_and_Inertial_Data,
                             Ball_aerodynamic_forces)


def test_Geometric_Data():

    # Test with default r=0.111
    r_expected = 0.111
    S_circle_expected = 0.03870756308
    S_sphere_expected = 0.1548302523
    Vol_expected = 0.005728719337

    r, S_circle, S_sphere, Vol = Geometric_Data(0.111)

    assert_almost_equal(r, r_expected)
    assert_almost_equal(S_circle, S_circle_expected)
    assert_almost_equal(S_sphere, S_sphere_expected)
    assert_almost_equal(Vol, Vol_expected)


def test_Mass_and_Inertial_Data():

    # Test with default r=0.111 (m) and mass = 0.440 (kg)
    I_matrix_expected = np.diag([1.0, 1.0, 1.0])
    I_matrix_expected *= 0.00361416

    I_matrix = Mass_and_Inertial_Data(0.111, 0.440)

    assert_array_almost_equal(I_matrix, I_matrix_expected)

