# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:32:11 2016

@author: Juan
"""
import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal)


from pyfme.utils.change_euler_quaternion import (quatern2euler, euler2quatern,
                                                 check_unitnorm,
                                                 vel_quaternion)


def test_quatern2euler():

    quaternion = np.array([0.8660254037844387, 0, 0.5, 0])

    euler_angles_expected = np.array([1.04719755, 0.0, 0.0])

    euler_angles = quatern2euler(quaternion)

    assert_array_almost_equal(euler_angles, euler_angles_expected)


def test_euler2quatern():

    euler_angles = np.array([0.0, 1.0471975511965976, 0.0])

    quaternion_expected = np.array([0.8660254037844387, 0.5, 0, 0])

    quaternion = euler2quatern(euler_angles)

    assert_array_almost_equal(quaternion, quaternion_expected)


def test_vel_quaternion():
    # test for bank velocity p = 1
    quaternion = np.array([0.8660254037844387, 0, 0.5, 0])

    ang_vel = np.array([1, 0, 0])

    d_quaternion_expected = np.array([0, 0.4330127018922193, 0, -0.25])

    d_quaternion = vel_quaternion(quaternion, ang_vel)

    assert_array_almost_equal(d_quaternion, d_quaternion_expected)

    # test for pitch velocity q = 1
    quaternion = np.array([0.8660254037844387, 0, 0.5, 0])

    ang_vel = np.array([0, 1, 0])

    d_quaternion_expected = np.array([-0.25, 0, 0.4330127018922193, 0])

    d_quaternion = vel_quaternion(quaternion, ang_vel)

    assert_array_almost_equal(d_quaternion, d_quaternion_expected)

    # test for yaw velocity r = 1
    quaternion = np.array([0.8660254037844387, 0, 0.5, 0])

    ang_vel = np.array([0, 0, 1])

    d_quaternion_expected = np.array([0, 0.25, 0, 0.4330127018922193])

    d_quaternion = vel_quaternion(quaternion, ang_vel)

    assert_array_almost_equal(d_quaternion, d_quaternion_expected)


def test_check_unitnorm():

    wrong_value = ([1, 0, 0.25, 0.5])

    with pytest.raises(ValueError):
        check_unitnorm(wrong_value)
