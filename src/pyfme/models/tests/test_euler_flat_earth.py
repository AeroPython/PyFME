# -*- coding: utf-8 -*-
"""
Tests of equations of euler flat earth model.
"""


import pytest
import numpy as np

from pyfme.models.euler_flat_earth import (linear_and_angular_momentum_eqs,
                                           jac_linear_and_angular_momentum_eqs,
                                           kinematic_angular_eqs,
                                           jac_kinematic_angular_eqs,
                                           navigation_eqs)


def test1_linear_and_angular_momentum_eqs():

    time = 0
    vel = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    mass = 10
    inertia = np.array([[1000,    0, -100],
                        [   0,  100,    0],
                        [-100,    0,  100]], dtype=float)
    forces = np.array([100., 100., 100.], dtype=float)
    moments = np.array([100., 1000., 100], dtype=float)

    expected_sol = np.array([10, 10, 10, 11./9, 1, 92./9], dtype=float)
    sol = linear_and_angular_momentum_eqs(time, vel, mass, inertia, forces,
                                          moments)
    assert(np.allclose(expected_sol, sol))


def test2_linear_and_angular_momentum_eqs():

    time = 0
    vel = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    mass = 10
    inertia = np.array([[100,    0, -10],
                        [  0,  100,   0],
                        [-10,    0, 100]], dtype=float)
    forces = np.array([1000, 10, 10], dtype=float)
    moments = np.array([100, 100, 100], dtype=float)

    expected_sol = np.array([99, 1, 2, 10./9, 1, 10./9], dtype=float)
    sol = linear_and_angular_momentum_eqs(time, vel, mass, inertia, forces,
                                          moments)
    assert(np.allclose(expected_sol, sol))


def test1_jac_linear_and_angular_momentum_eqs():

    time = 0
    vel = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    mass = 10
    inertia = np.array([[1000,    0, -100],
                        [   0,  100,    0],
                        [-100,    0,  100]], dtype=float)

    expected_sol = np.zeros([6, 6], dtype=float)

    expected_sol[0, 1] = 1
    expected_sol[0, 2] = - 1
    expected_sol[0, 4] = - 1
    expected_sol[0, 5] = 1

    expected_sol[1, 0] = - 1
    expected_sol[1, 2] = 1
    expected_sol[1, 3] = 1
    expected_sol[1, 5] = - 1

    expected_sol[2, 0] = 1
    expected_sol[2, 1] = - 1
    expected_sol[2, 3] = - 1
    expected_sol[2, 4] = 1

    expected_sol[3, 3] = 10./9
    expected_sol[3, 4] = 1
    expected_sol[3, 5] = - 1./9

    expected_sol[4, 3] = - 11
    expected_sol[4, 5] = - 7

    expected_sol[5, 3] = 91./9
    expected_sol[5, 4] = 9
    expected_sol[5, 5] = - 10./9

    sol = jac_linear_and_angular_momentum_eqs(time, vel, mass, inertia)

    assert(np.allclose(expected_sol, sol))


def test2_jac_linear_and_angular_momentum_eqs():

    time = 0
    vel = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    mass = 10
    inertia = np.array([[100,    0, -10],
                        [  0,  100,   0],
                        [-10,    0, 100]], dtype=float)

    expected_sol = np.zeros([6, 6], dtype=float)

    expected_sol[0, 2] = - 1
    expected_sol[0, 4] = - 1

    expected_sol[1, 3] = 1
    expected_sol[1, 5] = - 1

    expected_sol[2, 0] = 1
    expected_sol[2, 4] = 1

    expected_sol[3, 3] = 10./99
    expected_sol[3, 5] = - 1./99

    expected_sol[5, 3] = 1./99
    expected_sol[5, 5] = - 10./99

    sol = jac_linear_and_angular_momentum_eqs(time, vel, mass, inertia)

    assert(np.allclose(expected_sol, sol))


def test1_kinematic_angular_eqs():

    time = 0
    euler_angles = np.array([np.pi / 4, np.pi / 4, 0])
    ang_vel = np.array([1, 1, 1], dtype=float)

    expected_sol = np.array([0, 1 + 2 ** 0.5, 2])
    sol = kinematic_angular_eqs(time, euler_angles, ang_vel)

    assert(np.allclose(expected_sol, sol))


def test2_kinematic_angular_eqs():

    time = 0
    euler_angles = np.array([0, np.pi / 2, 0])
    ang_vel = np.array([0, 1, 0], dtype=float)

    expected_sol = np.array([0, 0, 1], dtype=float)
    sol = kinematic_angular_eqs(time, euler_angles, ang_vel)

    assert(np.allclose(expected_sol, sol))


def test1_jac_kinematic_angular_eqs():

    time = 0
    euler_angles = np.array([np.pi / 4, np.pi / 4, 0])
    ang_vel = np.array([1, 1, 1], dtype=float)

    expected_sol = np.zeros([3, 3])
    expected_sol[0, 1] = - 2 ** 0.5
    expected_sol[1, 0] = 2 * 2 ** 0.5
    expected_sol[2, 0] = 2

    sol = jac_kinematic_angular_eqs(time, euler_angles, ang_vel)

    assert(np.allclose(expected_sol, sol))


def test2_jac_kinematic_angular_eqs():

    time = 0
    euler_angles = np.array([0, np.pi / 2, 0])
    ang_vel = np.array([0, 1, 0], dtype=float)

    expected_sol = np.zeros([3, 3], dtype=float)
    expected_sol[0, 1] = - 1
    expected_sol[1, 0] = 1

    sol = jac_kinematic_angular_eqs(time, euler_angles, ang_vel)

    assert(np.allclose(expected_sol, sol))


def test1_navigation_eqs():

    time = 0
    lin_vel = np.array([1, 1, 1], dtype=float)
    euler_angles = np.array([np.pi / 4, np.pi / 4, 0])

    expected_sol = np.array([1 + (2 ** 0.5) / 2, 0, 1 - (2 ** 0.5) / 2])
    sol = navigation_eqs(time, lin_vel, euler_angles)

    assert(np.allclose(expected_sol, sol))


def test2_navigation_eqs():

    time = 0
    lin_vel = np.array([1, 0, 1], dtype=float)
    euler_angles = np.array([0, np.pi / 2, 0])

    expected_sol = np.array([1, - 1, 0], dtype=float)
    sol = navigation_eqs(time, lin_vel, euler_angles)

    assert(np.allclose(expected_sol, sol))