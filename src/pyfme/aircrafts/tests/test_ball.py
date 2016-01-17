# -*- coding: utf-8 -*-
"""
Frames of Reference orientation test functions
"""
import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)


from pyfme.aircrafts.ball import (geometric_data, mass_and_inertial_data,
                                  get_aerodynamic_forces,
                                  check_reynolds_number, check_sn,
                                  get_magnus_effect_forces)


def test_geometric_data():

    # Test with default r=0.111
    r_expected = 0.111
    S_circle_expected = 0.03870756308
    S_sphere_expected = 0.1548302523
    Vol_expected = 0.005728719337

    r, S_circle, S_sphere, Vol = geometric_data(0.111)

    assert_almost_equal(r, r_expected)
    assert_almost_equal(S_circle, S_circle_expected)
    assert_almost_equal(S_sphere, S_sphere_expected)
    assert_almost_equal(Vol, Vol_expected)


def test_mass_and_inertial_data():

    # Test with default r=0.111 (m) and mass = 0.440 (kg)
    inertia_expected = np.diag([1.0, 1.0, 1.0])
    inertia_expected *= 0.00361416

    inertia = mass_and_inertial_data(0.111, 0.440)

    assert_array_almost_equal(inertia, inertia_expected)


def test_check_reynolds_number():

    wrong_values = (0, 1e10)

    for value in wrong_values:
        Re = value
        with pytest.raises(ValueError):
            check_reynolds_number(Re)


def test_check_sn():

    wrong_values = (-0.5, 0.5)

    for value in wrong_values:
        Sn = value
        with pytest.raises(ValueError):
            check_sn(Sn)


def test_get_aerodynamic_forces():

    # Test with magnus effect
    lin_vel = np.array([30, 0, 0])
    ang_vel = np.array([0, 1, 1])
    TAS = 30
    rho = 1.225000018124288
    alpha = 0
    beta = 0
    magnus_effect = True
    forces = get_aerodynamic_forces(lin_vel, ang_vel, TAS, rho, alpha, beta,
                                    magnus_effect)

    Total_aerodynamic_forces_body = np.array([-10.83340379, 0.1973722863,
                                              -0.1973722863])

    assert_array_almost_equal(forces, Total_aerodynamic_forces_body)

    # Test without magnus effect
    lin_vel = np.array([30, 0, 0])
    ang_vel = np.array([0, 1, 1])
    TAS = 30
    rho = 1.225000018124288
    alpha = 0
    beta = 0
    magnus_effect = False
    forces = get_aerodynamic_forces(lin_vel, ang_vel, TAS, rho, alpha, beta,
                                    magnus_effect)
    Total_aerodynamic_forces_body = np.array([-10.83340379, 0,
                                              0])

    assert_array_almost_equal(forces, Total_aerodynamic_forces_body)


def test_get_magnus_effect_forces():

    lin_vel = np.array([30, 0, 0])
    ang_vel = np.array([0, 1, 1])
    TAS = 30
    rho = 1.225000018124288
    radius = 0.111
    A_front = 0.03870756308
    alpha = 0
    beta = 0

    F_magnus_vector_body_expected = np.array([0, 0.1973722863, -0.1973722863])
    forces = get_magnus_effect_forces(lin_vel, ang_vel, TAS, rho, radius,
                                      A_front, alpha, beta)
    assert_array_almost_equal(forces, F_magnus_vector_body_expected)
