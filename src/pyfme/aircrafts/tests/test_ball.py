# -*- coding: utf-8 -*-
"""
Frames of Reference orientation test functions
"""
import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)


from pyfme.ball.ball import (Geometric_Data, Mass_and_Inertial_Data,
                             Ball_aerodynamic_forces, Check_Reynolds_number,
                             Check_Sn)


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


def test_Check_Reynolds_number():

    wrong_values = (0, 1e10)

    for value in wrong_values:
        Re = value
        with pytest.raises(ValueError):
            Check_Reynolds_number(Re)


def test_Check_Sn():

    wrong_values = (-0.5, 0.5)

    for value in wrong_values:
        Sn = value
        with pytest.raises(ValueError):
            Check_Sn(Sn)


def test_Ball_aerodynamic_forces():

    # Test with a pitch rotation
    velocity_vector = np.array([30, 0, 0, 0, 1, 1])
    h = 0
    alpha = 0
    beta = 0

    forces = Ball_aerodynamic_forces(velocity_vector, h, alpha,
                                     beta)
    Cd_expected = 0.5077155812
    D_expected = 10.83340379
    D_vector_body_expected = np.array([-10.83340379, 0, 0])
    C_magnus_expected = 0.01308147545
    F_magnus_expected = 0.2791265641399841
    F_magnus_vector_body_expected = np.array([0., -0.1973722863,
                                              0.1973722863])

    assert_array_almost_equal(forces[0], Cd_expected)
    assert_array_almost_equal(forces[1], D_expected)
    assert_array_almost_equal(forces[2], D_vector_body_expected)
    assert_array_almost_equal(forces[3], C_magnus_expected)
    assert_array_almost_equal(forces[4], F_magnus_expected)
    assert_array_almost_equal(forces[5], F_magnus_vector_body_expected)
