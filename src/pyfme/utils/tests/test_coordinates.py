# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Frames of Reference orientation test functions
----------------------------------------------
"""
import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal)


from pyfme.utils.coordinates import (body2hor, hor2body,
                                     check_theta_phi_psi_range,
                                     hor2wind, wind2hor,
                                     check_gamma_mu_chi_range,
                                     body2wind, wind2body,
                                     check_alpha_beta_range)


def test_check_theta_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_theta_phi_psi_range(value, 0, 0)
        assert ("ValueError: Theta value is not inside correct range"
                in excinfo.exconly())


def test_check_phi_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_theta_phi_psi_range(0, value, 0)
        assert ("ValueError: Phi value is not inside correct range"
                in excinfo.exconly())


def test_check_psi_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_theta_phi_psi_range(0, 0, value)
        assert ("ValueError: Psi value is not inside correct range"
                in excinfo.exconly())


def test_body2hor():

    # Test with a pitch rotation
    vector_body = np.array([1, 1, 1])
    theta, phi, psi = np.deg2rad(45), 0, 0

    vector_hor = body2hor(vector_body, theta, phi, psi)

    vector_hor_expected = np.array([2 * 0.70710678118654757, 1, 0])

    assert_array_almost_equal(vector_hor, vector_hor_expected)

    # Test with a roll rotation
    vector_body = np.array([1, 1, 1])
    theta, phi, psi = 0, np.deg2rad(45), 0

    vector_hor = body2hor(vector_body, theta, phi, psi)

    vector_hor_expected = np.array([1, 0, 2 * 0.70710678118654757])

    assert_array_almost_equal(vector_hor, vector_hor_expected)

    # Test with a yaw rotation
    vector_body = np.array([1, 1, 1])
    theta, phi, psi = 0, 0, np.deg2rad(45)

    vector_hor = body2hor(vector_body, theta, phi, psi)

    vector_hor_expected = np.array([0, 2 * 0.70710678118654757, 1])

    assert_array_almost_equal(vector_hor, vector_hor_expected)


def test_hor2body():

    # Test with a pitch rotation
    vector_hor = np.array([2 * 0.70710678118654757,  1, 0])
    theta, phi, psi = np.deg2rad(45), 0, 0

    vector_body_expected = np.array([1, 1, 1])

    vector_body = hor2body(vector_hor, theta, phi, psi)

    assert_array_almost_equal(vector_body, vector_body_expected)

    # Test with a roll rotation
    vector_hor = np.array([1, 0, 2 * 0.70710678118654757])
    theta, phi, psi = 0, np.deg2rad(45), 0

    vector_body_expected = np.array([1, 1, 1])

    vector_body = hor2body(vector_hor, theta, phi, psi)

    assert_array_almost_equal(vector_body, vector_body_expected)

    # Test with a yaw rotation
    vector_hor = np.array([0, 2 * 0.70710678118654757, 1])
    theta, phi, psi = 0, 0, np.deg2rad(45)

    vector_body_expected = np.array([1, 1, 1])

    vector_body = hor2body(vector_hor, theta, phi, psi)

    assert_array_almost_equal(vector_body, vector_body_expected)


def test_check_gamma_mu_chi_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        angles = [0, 0, 0]
        for ii in range(3):
            angles[ii] = value
            with pytest.raises(ValueError):
                check_gamma_mu_chi_range(*angles)


def test_check_gamma_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_gamma_mu_chi_range(value, 0, 0)
        assert ("ValueError: Gamma value is not inside correct range"
                in excinfo.exconly())


def test_check_mu_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_gamma_mu_chi_range(0, value, 0)
        assert ("ValueError: Mu value is not inside correct range"
                in excinfo.exconly())


def test_check_chi_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_gamma_mu_chi_range(0, 0, value)
        assert ("ValueError: Chi value is not inside correct range"
                in excinfo.exconly())


def test_wind2hor():

    # Test with a pitch rotation
    vector_wind = np.array([1, 1, 1])
    gamma, mu, chi = np.deg2rad(45), 0, 0

    vector_hor = wind2hor(vector_wind, gamma, mu, chi)

    vector_hor_expected = np.array([2 * 0.70710678118654757, 1, 0])

    assert_array_almost_equal(vector_hor, vector_hor_expected)

    # Test with a roll rotation
    vector_wind = np.array([1, 1, 1])
    gamma, mu, chi = 0, np.deg2rad(45), 0

    vector_hor = wind2hor(vector_wind, gamma, mu, chi)

    vector_hor_expected = np.array([1, 0, 2 * 0.70710678118654757])

    assert_array_almost_equal(vector_hor, vector_hor_expected)

    # Test with a yaw rotation
    vector_wind = np.array([1, 1, 1])
    gamma, mu, chi = 0, 0, np.deg2rad(45)

    vector_hor = wind2hor(vector_wind, gamma, mu, chi)

    vector_hor_expected = np.array([0, 2 * 0.70710678118654757, 1])

    assert_array_almost_equal(vector_hor, vector_hor_expected)


def test_hor2wind():

    # Test with a pitch rotation
    vector_hor = np.array([2 * 0.70710678118654757,  1, 0])
    gamma, mu, chi = np.deg2rad(45), 0, 0

    vector_wind_expected = np.array([1, 1, 1])

    vector_wind = hor2wind(vector_hor, gamma, mu, chi)

    assert_array_almost_equal(vector_wind, vector_wind_expected)

    # Test with a roll rotation
    vector_hor = np.array([1, 0, 2 * 0.70710678118654757])
    gamma, mu, chi = 0, np.deg2rad(45), 0

    vector_wind_expected = np.array([1, 1, 1])

    vector_wind = hor2wind(vector_hor, gamma, mu, chi)

    assert_array_almost_equal(vector_wind, vector_wind_expected)

    # Test with a yaw rotation
    vector_hor = np.array([0, 2 * 0.70710678118654757, 1])
    gamma, mu, chi = 0, 0, np.deg2rad(45)

    vector_wind_expected = np.array([1, 1, 1])

    vector_wind = hor2wind(vector_hor, gamma, mu, chi)

    assert_array_almost_equal(vector_wind, vector_wind_expected)


def test_check_alpha_beta_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        angles = [0, 0]
        for ii in range(2):
            angles[ii] = value
            with pytest.raises(ValueError):
                check_alpha_beta_range(*angles)


def test_check_alpha_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_alpha_beta_range(value, 0)
        assert ("ValueError: Alpha value is not inside correct range"
                in excinfo.exconly())


def test_check_beta_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        with pytest.raises(ValueError) as excinfo:
            check_alpha_beta_range(0, value)
        assert ("ValueError: Beta value is not inside correct range"
                in excinfo.exconly())


def test_wind2body():

    # Test with an increment of the angle of attack
    vector_wind = np.array([1, 1, 1])
    alpha, beta = np.deg2rad(45), 0

    vector_body = wind2body(vector_wind, alpha, beta)

    vector_body_expected = np.array([0, 1, 2 * 0.70710678118654757])

    assert_array_almost_equal(vector_body, vector_body_expected)

    # Test with an increment of the sideslip angle
    vector_wind = np.array([1, 1, 1])
    alpha, beta = 0, np.deg2rad(45)

    vector_body = wind2body(vector_wind, alpha, beta)

    vector_body_expected = np.array([0, 2 * 0.70710678118654757, 1])

    assert_array_almost_equal(vector_body, vector_body_expected)


def test_body2wind():

    # Test with an increment of the angle of attack
    vector_body = np.array([0, 1, 2 * 0.70710678118654757])
    alpha, beta = np.deg2rad(45), 0

    vector_wind = body2wind(vector_body, alpha, beta)

    vector_wind_expected = np.array([1, 1, 1])

    assert_array_almost_equal(vector_wind, vector_wind_expected)

    # Test with an increment of the sideslip angle
    vector_body = np.array([0, 2 * 0.70710678118654757, 1])
    alpha, beta = 0, np.deg2rad(45)

    vector_wind = body2wind(vector_body, alpha, beta)

    vector_wind_expected = np.array([1, 1, 1])

    assert_array_almost_equal(vector_wind, vector_wind_expected)
