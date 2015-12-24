# -*- coding: utf-8 -*-
"""
Frames of Reference orientation test functions
"""
import pytest

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal)


from pyfme.utils.coordinates import (body2hor, hor2body,
                                     check_theta_phi_psi_range)


def test_check_theta_phi_psi_range():

    wrong_values = (3 * np.pi, - 3 * np.pi)

    for value in wrong_values:
        # 0 is always a correct value
        angles = [0, 0, 0]
        for ii in range(3):
            angles[ii] = value
            with pytest.raises(ValueError):
                check_theta_phi_psi_range(*angles)


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
