# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Inputs Generator Tests
----------------------
Test functions for input generator module.

"""
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from pyfme.utils.input_generator import (step,
                                         doublet,
                                         sinusoid,
                                         ramp,
                                         harmonic)


def test_step():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:6] = A

    step_input = step(t_init, T, A, time, offset=0, var=None)

    assert_almost_equal(step_input, expected_input)


def test_step_bounds_not_included():
    t_init = 0.1
    T = 4.8
    A = 1.5
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[1:5] = A

    step_input = step(t_init, T, A, time, offset=0, var=None)

    assert_almost_equal(step_input, expected_input)


def test_step_offset():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    offset = 2.6

    expected_input = np.zeros([11])
    expected_input[0:6] += A + offset

    step_input = step(t_init, T, A, time, offset=offset, var=None)

    assert_almost_equal(step_input, expected_input)


def test_step_var():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var.copy()
    expected_input[0:6] += A

    step_input = step(t_init, T, A, time, offset=0, var=var)

    assert_almost_equal(step_input, expected_input)


def test_step_var_and_offset():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    offset = 2.6
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var
    expected_input[0:6] += A + offset

    step_input = step(t_init, T, A, time, offset=offset, var=var)

    assert_almost_equal(step_input, expected_input)


def test_step_wrong_size_var():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones([10])

    with pytest.raises(ValueError) as excinfo:
        step(t_init, T, A, time, offset=0, var=var)
    assert ("ValueError: var and time must have the same size"
            in excinfo.exconly())


def test_step_wrong_not_scalar_offset():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    offset = var

    with pytest.raises(TypeError) as excinfo:
        step(t_init, T, A, time, offset=offset)


def test_doublet():
    t_init = 0
    T = 5.
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:3] = A/2
    expected_input[3:6] = -A/2

    doublet_input = doublet(t_init, T, A, time, offset=0, var=None)

    assert_almost_equal(doublet_input, expected_input)


def test_doublet_bounds_not_included():
    t_init = 0.1
    T = 4.8
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[1:3] = A/2
    expected_input[3:5] = -A/2

    doublet_input = doublet(t_init, T, A, time, offset=0, var=None)

    assert_almost_equal(doublet_input, expected_input)


def test_doublet_offset():
    t_init = 0
    T = 5
    A = 1.5
    time = np.linspace(0, 10, 11)
    offset = 2.6

    expected_input = np.zeros([11])
    expected_input[0:3] += A/2 + offset
    expected_input[3:6] += -A/2 + offset

    doublet_input = doublet(t_init, T, A, time, offset=offset, var=None)

    assert_almost_equal(doublet_input, expected_input)


def test_doublet_var():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var.copy()
    expected_input[0:3] += A/2
    expected_input[3:6] += -A/2

    doublet_input = doublet(t_init, T, A, time, offset=0, var=var)

    assert_almost_equal(doublet_input, expected_input)


def test_doublet_var_and_offset():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    offset = 2.6
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var
    expected_input[0:3] += A/2 + offset
    expected_input[3:6] += -A/2 + offset

    doublet_input = doublet(t_init, T, A, time, offset=offset, var=var)

    assert_almost_equal(doublet_input, expected_input)


def test_doublet_wrong_size_var():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones([10])

    with pytest.raises(ValueError) as excinfo:
        doublet(t_init, T, A, time, offset=0, var=var)
    assert ("ValueError: var and time must have the same size"
            in excinfo.exconly())


def test_doublet_wrong_not_scalar_offset():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    offset = var

    with pytest.raises(TypeError) as excinfo:
        doublet(t_init, T, A, time, offset=offset)


def test_ramp():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:5] = np.array([0, A/4, A/2, 3*A/4, A])

    ramp_input = ramp(t_init, T, A, time, offset=0, var=None)

    assert_almost_equal(ramp_input, expected_input)


def test_ramp_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 1

    expected_input = np.zeros([11])
    expected_input[0:5] += np.array([0, A/4, A/2, 3*A/4, A]) + offset

    ramp_input = ramp(t_init, T, A, time, offset=offset, var=None)

    assert_almost_equal(ramp_input, expected_input)


def test_ramp_var():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var.copy()
    expected_input[0:5] += np.array([0, A/4, A/2, 3*A/4, A])

    ramp_input = ramp(t_init, T, A, time, offset=0, var=var)

    assert_almost_equal(ramp_input, expected_input)


def test_ramp_var_and_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 2
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var
    expected_input[0:5] += np.array([0, A/4, A/2, 3*A/4, A]) + offset

    ramp_input = ramp(t_init, T, A, time, offset=offset, var=var)

    assert_almost_equal(ramp_input, expected_input)


def test_ramp_wrong_size_var():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones([10])

    with pytest.raises(ValueError) as excinfo:
        doublet(t_init, T, A, time, offset=0, var=var)
    assert ("ValueError: var and time must have the same size"
            in excinfo.exconly())


def test_ramp_wrong_not_scalar_offset():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    offset = var

    with pytest.raises(TypeError) as excinfo:
        ramp(t_init, T, A, time, offset=offset)


def test_sinusoid():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)

    expected_input = np.zeros([11])
    expected_input[0:5] = np.array([0, A/2, 0, -A/2, 0])

    sinusoid_input = sinusoid(t_init, T, A, time)

    assert_almost_equal(sinusoid_input, expected_input)


def test_sinusoid_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 1

    expected_input = np.zeros([11])
    expected_input[0:5] += np.array([0, A/2, 0, -A/2, 0]) + offset

    sinusoid_input = sinusoid(t_init, T, A, time, offset=offset)

    assert_almost_equal(sinusoid_input, expected_input)


def test_sinusoid_phase():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    phase = np.pi/2

    expected_input = np.zeros([11])
    expected_input[0:5] += np.array([A/2, 0, -A/2, 0, A/2])

    sinusoid_input = sinusoid(t_init, T, A, time, phase=phase)

    assert_almost_equal(sinusoid_input, expected_input)


def test_sinusoid_var():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var.copy()
    expected_input[0:5] += np.array([0, A/2, 0, -A/2, 0])

    sinusoid_input = sinusoid(t_init, T, A, time, offset=0, var=var)

    assert_almost_equal(sinusoid_input, expected_input)


def test_sinusoid_var_and_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 2
    var = np.ones_like(time)
    var[0::2] = -1

    expected_input = var
    expected_input[0:5] += np.array([0, A/2, 0, -A/2, 0]) + offset

    sinusoid_input = sinusoid(t_init, T, A, time, offset=offset, var=var)

    assert_almost_equal(sinusoid_input, expected_input)


def test_sinusoid_wrong_size_var():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones([10])

    with pytest.raises(ValueError) as excinfo:
        sinusoid(t_init, T, A, time, offset=0, var=var)
    assert ("ValueError: var and time must have the same size"
            in excinfo.exconly())


def test_sinusoid_wrong_not_scalar_offset():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    offset = var

    with pytest.raises(TypeError) as excinfo:
        sinusoid(t_init, T, A, time, offset=offset)


def test_harmonic():
    t_init = 0
    T = 4.
    A = 3.0
    time = np.linspace(0, 10, 11)
    f = 0.25

    expected_input = np.zeros([11])
    expected_input[0:5] = np.array([0, A/2, 0, -A/2, 0])

    harmonic_input = harmonic(t_init, T, A, time, f, phase=0,
                              offset=0, var=None)

    assert_almost_equal(harmonic_input, expected_input)


def test_harmonic_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 1
    f = 0.25

    expected_input = np.zeros([11])
    expected_input[0:5] += np.array([0, A/2, 0, -A/2, 0]) + offset

    harmonic_input = harmonic(t_init, T, A, time, f, offset=offset)

    assert_almost_equal(harmonic_input, expected_input)


def test_harmonic_phase():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    phase = np.pi/2
    f = 0.25

    expected_input = np.zeros([11])
    expected_input[0:5] += np.array([A/2, 0, -A/2, 0, A/2])

    harmonic_input = harmonic(t_init, T, A, time, f, phase=phase)

    assert_almost_equal(harmonic_input, expected_input)


def test_harmonic_var():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    var[0::2] = -1
    f = 0.25

    expected_input = var.copy()
    expected_input[0:5] += np.array([0, A/2, 0, -A/2, 0])

    harmonic_input = harmonic(t_init, T, A, time, f, offset=0, var=var)

    assert_almost_equal(harmonic_input, expected_input)


def test_harmonic_var_and_offset():
    t_init = 0
    T = 4.
    A = 3.
    time = np.linspace(0, 10, 11)
    offset = 2
    var = np.ones_like(time)
    var[0::2] = -1
    f = 0.25

    expected_input = var
    expected_input[0:5] += np.array([0, A/2.0, 0, -A/2, 0]) + offset

    harmonic_input = harmonic(t_init, T, A, time, f, offset=offset, var=var)

    assert_almost_equal(harmonic_input, expected_input)


def test_harmonic_wrong_size_var():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones([10])
    f = 0.25

    with pytest.raises(ValueError) as excinfo:
        harmonic(t_init, T, A, time, f, offset=0, var=var)
    assert ("ValueError: var and time must have the same size"
            in excinfo.exconly())


def test_harmonic_wrong_not_scalar_offset():
    t_init = 0
    T = 5.
    A = 1.5
    time = np.linspace(0, 10, 11)
    var = np.ones_like(time)
    offset = var
    f = 0.25

    with pytest.raises(TypeError) as excinfo:
        harmonic(t_init, T, A, time, f, offset=offset)
