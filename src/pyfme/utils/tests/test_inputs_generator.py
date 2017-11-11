# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Inputs Generator Tests
----------------------
Test functions for input generator module.

"""
import numpy as np
from numpy.testing import assert_almost_equal
from pyfme.utils.input_generator import (Step, Doublet, Ramp, Harmonic,
                                         Constant)


def test_input_scalar_output_scalar():
    control = Constant(1.5)
    control_value = control(1.5)

    assert isinstance(control_value, float)


def test_step():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:6] = A

    step_input = Step(t_init, T, A, offset=0)
    real_input = step_input(time)

    assert_almost_equal(real_input, expected_input)


def test_step_bounds_not_included():
    t_init = 0.1
    T = 4.8
    A = 1.5
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[1:5] = A

    step_input = Step(t_init, T, A, offset=0)
    real_input = step_input(time)

    assert_almost_equal(real_input, expected_input)


def test_step_offset():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    offset = 2.6

    expected_input = np.zeros([11]) + offset
    expected_input[0:6] += A

    step_input = Step(t_init, T, A, offset=offset)
    real_input = step_input(time)

    assert_almost_equal(real_input, expected_input)


def test_doublet():
    t_init = 0
    T = 5.
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:3] = A/2
    expected_input[3:6] = -A/2

    doublet_input = Doublet(t_init, T, A, offset=0)
    real_input = doublet_input(time)

    assert_almost_equal(real_input, expected_input)


def test_doublet_bounds_not_included():
    t_init = 0.1
    T = 4.8
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[1:3] = A/2
    expected_input[3:5] = -A/2

    doublet_input = Doublet(t_init, T, A, offset=0)
    real_input = doublet_input(time)

    assert_almost_equal(real_input, expected_input)


def test_doublet_offset():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    offset = 2.6

    expected_input = np.zeros([11]) + offset
    expected_input[0:3] += A/2
    expected_input[3:6] += -A/2

    doublet_input = Doublet(t_init, T, A, offset=offset)
    real_input = doublet_input(time)

    assert_almost_equal(real_input, expected_input)


def test_ramp():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:5] = np.array([0, A/4, A/2, 3*A/4, A])

    ramp_input = Ramp(t_init, T, A, offset=0)
    real_input = ramp_input(time)

    assert_almost_equal(real_input, expected_input)


def test_ramp_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 1

    expected_input = np.zeros([11]) + offset
    expected_input[0:5] += np.array([0, A/4, A/2, 3*A/4, A])

    ramp_input = Ramp(t_init, T, A, offset=offset)
    real_input = ramp_input(time)

    assert_almost_equal(real_input, expected_input)


def test_harmonic():
    t_init = 0
    T = 4.
    A = 3.0
    time = np.linspace(0, 10, 11)
    f = 0.25

    expected_input = np.zeros([11])
    expected_input[0:5] = np.array([0, A/2, 0, -A/2, 0])

    harmonic_input = Harmonic(t_init, T, A, f, phase=0, offset=0)
    real_input = harmonic_input(time)

    assert_almost_equal(real_input, expected_input)


def test_harmonic_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 1
    f = 0.25

    expected_input = np.zeros([11]) + offset
    expected_input[0:5] += np.array([0, A/2, 0, -A/2, 0])

    harmonic_input = Harmonic(t_init, T, A, f, phase=0, offset=offset)
    real_input = harmonic_input(time)

    assert_almost_equal(real_input, expected_input)


def test_harmonic_phase():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    phase = np.pi/2
    f = 0.25

    expected_input = np.zeros([11])
    expected_input[0:5] += np.array([A/2, 0, -A/2, 0, A/2])

    harmonic_input = Harmonic(t_init, T, A, f, phase=phase)
    real_input = harmonic_input(time)

    assert_almost_equal(real_input, expected_input)


def test_constant():

    offset = 3.5
    time = np.linspace(0, 5, 11)

    expected_input = np.full_like(time, offset)

    constant = Constant(offset)
    real_input = constant(time)

    assert_almost_equal(real_input, expected_input)


def test_add_controls_01():

    offset1 = 3.5
    offset2 = 1.2

    time = np.linspace(0, 5, 11)

    expected_input = np.full_like(time, offset1 + offset2)

    constant1 = Constant(offset1)
    constant2 = Constant(offset2)

    constant_input = constant1 + constant2
    real_input = constant_input(time)

    assert_almost_equal(real_input, expected_input)


def test_add_controls_02():

    time = np.linspace(0, 10, 11)

    # Define harmonic input
    t_init = 0
    T = 4.
    A = 3.
    phase = np.pi / 2
    f = 0.25

    expected_harm_input = np.zeros([11])
    expected_harm_input[0:5] += np.array([A / 2, 0, -A / 2, 0, A / 2])

    harmonic_input = Harmonic(t_init, T, A, f, phase=phase)

    # Define ramp input
    t_init = 0
    T = 4.
    A = 3.

    expected_ramp_input = np.zeros([11])
    expected_ramp_input[0:5] = np.array([0, A / 4, A / 2, 3 * A / 4, A])

    ramp_input = Ramp(t_init, T, A, offset=0)

    # Add both
    composed_input = ramp_input + harmonic_input
    real_input = composed_input(time)

    expected_input = expected_ramp_input + expected_harm_input

    assert_almost_equal(real_input, expected_input)


def test_subtract_controls_01():

    time = np.linspace(0, 10, 11)

    # Define harmonic input
    t_init = 0
    T = 4.
    A = 3.
    phase = np.pi / 2
    f = 0.25

    expected_harm_input = np.zeros([11])
    expected_harm_input[0:5] += np.array([A / 2, 0, -A / 2, 0, A / 2])

    harmonic_input = Harmonic(t_init, T, A, f, phase=phase)

    # Define ramp input
    t_init = 0
    T = 4.
    A = 3.

    expected_ramp_input = np.zeros([11])
    expected_ramp_input[0:5] = np.array([0, A / 4, A / 2, 3 * A / 4, A])

    ramp_input = Ramp(t_init, T, A, offset=0)

    # Subtract both
    composed_input = ramp_input - harmonic_input
    real_input = composed_input(time)

    expected_input = expected_ramp_input - expected_harm_input

    assert_almost_equal(real_input, expected_input)


def test_multiply_controls_01():

    time = np.linspace(0, 10, 11)

    # Define harmonic input
    t_init = 0
    T = 4.
    A = 3.
    phase = np.pi / 2
    f = 0.25

    expected_harm_input = np.zeros([11])
    expected_harm_input[0:5] += np.array([A / 2, 0, -A / 2, 0, A / 2])

    harmonic_input = Harmonic(t_init, T, A, f, phase=phase)

    # Define ramp input
    t_init = 0
    T = 4.
    A = 3.

    expected_ramp_input = np.zeros([11])
    expected_ramp_input[0:5] = np.array([0, A / 4, A / 2, 3 * A / 4, A])

    ramp_input = Ramp(t_init, T, A, offset=0)

    # Multiply both
    composed_input = ramp_input * harmonic_input
    real_input = composed_input(time)

    expected_input = expected_ramp_input * expected_harm_input

    assert_almost_equal(real_input, expected_input)

    time = np.linspace(0, 10, 11)

    # Define harmonic input
    t_init = 0
    T = 4.
    A = 3.
    phase = np.pi / 2
    f = 0.25

    expected_harm_input = np.zeros([11])
    expected_harm_input[0:5] += np.array([A / 2, 0, -A / 2, 0, A / 2])

    harmonic_input = Harmonic(t_init, T, A, f, phase=phase)

    # Define ramp input
    t_init = 0
    T = 4.
    A = 3.

    expected_ramp_input = np.zeros([11])
    expected_ramp_input[0:5] = np.array([0, A / 4, A / 2, 3 * A / 4, A])

    ramp_input = Ramp(t_init, T, A, offset=0)

    # Add both
    composed_input = ramp_input + harmonic_input
    real_input = composed_input(time)

    expected_input = expected_harm_input + expected_ramp_input

    assert_almost_equal(real_input, expected_input)
