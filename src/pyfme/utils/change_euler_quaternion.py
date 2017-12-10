# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:34:43 2016

@author: Juan
"""

from math import atan2, asin, cos, sin
import numpy as np


def quatern2euler(quaternion):
    '''Given a quaternion, the euler_angles vector is returned.

    Parameters
    ----------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_1, q_2, q_3]

    Returns
    -------
    euler_angles : array_like
        1x3 array with the euler angles: [theta, phi, psi]    (rad)

    References
    ----------
    .. [1] "Modeling and Simulation of Aerospace Vehicle Dynamics" (Aiaa\
        Education Series) Peter H. Ziepfel
    '''
    check_unitnorm(quaternion)

    q_0, q_1, q_2, q_3 = quaternion

    check_unitnorm(quaternion)

    psi = atan2(2 * (q_1 * q_2 + q_0 * q_3),
                q_0 ** 2 + q_1 ** 2 - q_2 ** 2 - q_3 ** 2)

    theta = asin(-2 * (q_1 * q_3 - q_0 * q_2))

    phi = atan2(2 * (q_2 * q_3 + q_0 * q_1),
                q_0 ** 2 + q_1 ** 2 - q_2 ** 2 - q_3 ** 2)

    euler_angles = np.array([theta, phi, psi])

    return euler_angles


def euler2quatern(euler_angles):
    '''Given the euler_angles vector, the quaternion vector is returned.

    Parameters
    ----------
    euler_angles : array_like
        1x3 array with the euler angles: [theta, phi, psi]    (rad)

    Returns
    -------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_1, q_2, q_3]

    References
    ----------
    .. [1] "Modeling and Simulation of Aerospace Vehicle Dynamics" (Aiaa\
        Education Series) Peter H. Ziepfel
    '''
    theta, phi, psi = euler_angles

    q_0 = cos(psi / 2.) * cos(theta / 2.) * cos(phi / 2.) +\
        sin(psi / 2) * sin(theta / 2) * sin(phi / 2)

    q_1 = cos(psi / 2.) * cos(theta / 2.) * sin(phi / 2.) -\
        sin(psi / 2.) * sin(theta / 2.) * cos(phi / 2.)

    q_2 = cos(psi / 2) * sin(theta / 2.) * cos(phi / 2.) +\
        sin(psi / 2.) * cos(theta / 2.) * sin(phi / 2.)

    q_3 = sin(psi / 2.) * cos(theta / 2.) * cos(phi / 2.) -\
        cos(psi / 2.) * sin(theta / 2.) * sin(phi / 2.)

    quaternion = np.array([q_0, q_1, q_2, q_3])

    return quaternion


def vel_quaternion(quaternion, ang_vel):
    '''Given the angular velocity vector and the quaternion,
    the derivative of the quaternion vector is returned.

    Parameters
    ----------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion: q_0, q_1, q_2, q_3

    ang_vel : array_like
        (p, q, r) Absolute angular velocity of aircraft with respect to
        inertial frame of ref in body axes coordinates

    Returns
    -------
    d_quaternion : array_like
        1x4 vector with the derivative of the four elements of the quaternion:
        [d_q_0, d_q_1, d_q_2, d_q_3]

    References
    ----------
    .. [1] "Modeling and Simulation of Aerospace Vehicle Dynamics" (Aiaa\
        Education Series) Peter H. Ziepfel
    '''
    p, q, r = ang_vel

    q_0, q_1, q_2, q_3 = quaternion

    d_q_0 = 0.5 * (-p * q_1 - q * q_2 - r * q_3)

    d_q_1 = 0.5 * (p * q_0 + r * q_2 - q * q_3)

    d_q_2 = 0.5 * (q * q_0 - r * q_1 + p * q_3)

    d_q_3 = 0.5 * (r * q_0 + q * q_1 - p * q_2)

    d_quaternion = np.array([d_q_0, d_q_1, d_q_2, d_q_3])

    return d_quaternion


def check_unitnorm(quaternion):
    '''Given a quaternion, it checks the modulus (it must be unit). If it is
    not unit, it raises an error.

    Parameters
    ----------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_1, q_2, q_3]
    Raises
    ------
    ValueError:
        Selected quaternion norm is not unit
    '''
    q_0, q_1, q_2, q_3 = quaternion

    check_value = np.isclose([q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2], [1])

    if not check_value:
        raise ValueError('Selected quaternion norm is not unit')
