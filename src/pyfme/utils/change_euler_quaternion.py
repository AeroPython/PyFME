# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:34:43 2016

@author: Juan
"""

from math import atan2, asin, cos, sin
import numpy as np


def quatern2euler(q_0, q_1, q_2, q_3):

    check_orthogonality_numpyv_1_6(q_0, q_1, q_2, q_3)

    psi = atan2(2 * (q_1 * q_2 + q_0 * q_3),
                q_0 ** 2 + q_1 ** 2 - q_2 ** 2 - q_3 ** 2)

    theta = asin(-2 * (q_1 * q_3 - q_0 * q_2))

    phi = atan2(2 * (q_2 * q_3 + q_0 * q_1),
                q_0 ** 2 + q_1 ** 2 - q_2 ** 2 - q_3 ** 2)
    return psi, theta, phi


def euler2quatern(psi, theta, phi):

    q_0 = cos(psi / 2.) * cos(theta / 2.) * cos(phi / 2.) +\
        sin(psi / 2) * sin(theta / 2) * sin(phi / 2)

    q_1 = cos(psi / 2.) * cos(theta / 2.) * sin(phi / 2.) -\
        sin(psi / 2.) * sin(theta / 2.) * cos(phi / 2.)

    q_2 = cos(psi / 2) * sin(theta / 2.) * cos(phi / 2.) +\
        sin(psi / 2.) * cos(theta / 2.) * sin(phi / 2.)

    q_3 = sin(psi / 2.) * cos(theta / 2.) * cos(phi / 2.) -\
        cos(psi / 2.) * sin(theta / 2.) * sin(phi / 2.)

    return q_0, q_1, q_2, q_3


def check_orthogonality(q_0, q_1, q_2, q_3):

    check_value = np.isclose([q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2], [1])
    if not check_value:
        raise ValueError('selected quaternion is not orthogonal')


def check_orthogonality_numpyv_1_6(q_0, q_1, q_2, q_3):

    check_value = q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2
    if not 0.999999 <= check_value <= 1.000001:
        raise ValueError('selected quaternion is not orthogonal')


print(quatern2euler(0.8660254037844387, 0, 0.5, 0))

print(euler2quatern(0.0, 1.0471975511965976, 0.0))
