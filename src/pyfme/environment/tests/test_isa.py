# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Tests of the ISA functions
--------------------------
All numerical results are validated against the `COESA`_ standard.

.. _`COESA`: http://hdl.handle.net/2060/19770009539

Based on scikit-aero (c) 2012 scikit-aero authors.
"""
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal)

import pytest

from pyfme.environment.atmosphere import ISA1976

atmosphere = ISA1976()

def test_sea_level():
    h = 0.0  # m
    expected_T = 288.15  # K
    expected_p = 101325.0  # Pa
    expected_rho = 1.2250  # kg / m3
    expected_a = 340.293990543  # m / s

    T, p, rho, a = atmosphere(h)

    # Reads: "Assert if T equals expected_T"
    assert_equal(T, expected_T)
    assert_equal(p, expected_p)
    assert_almost_equal(rho, expected_rho, decimal=4)
    assert_almost_equal(a, expected_a, decimal=4)


def test_altitude_is_out_of_range():
    wrong_h = (-1.0, 84501)  # m

    for h in wrong_h:
        with pytest.raises(ValueError):
            atmosphere(h)


def test_results_under_11km():
    h = np.array([0.0,
                  50.0,
                  550.0,
                  6500.0,
                  10000.0,
                  11000.0
                  ])  # m
    expected_T = np.array([288.150,
                           287.825,
                           284.575,
                           245.900,
                           223.150,
                           216.650
                           ])  # K

    expected_rho = np.array([1.2250,
                             1.2191,
                             1.1616,
                             0.62384,
                             0.41271,
                             0.36392
                             ])  # kg / m3

    expected_a = np.array([340.29,
                           340.10,
                           338.18,
                           314.36,
                           299.46,
                           295.07
                           ])  # m / s

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=4)
        assert_almost_equal(a, expected_a[ii], decimal=2)


def test_results_under_20km():
    h = np.array([12000,
                  14200,
                  17500,
                  20000
                  ])  # m
    expected_T = np.array([216.650,
                           216.650,
                           216.650,
                           216.650,
                           ])  # K

    expected_rho = np.array([0.31083,
                             0.21971,
                             0.13058,
                             0.088035
                             ])  # kg / m3

    expected_a = np.array([295.07,
                           295.07,
                           295.07,
                           295.07,
                           ])

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=5)
        assert_almost_equal(a, expected_a[ii], decimal=2)


def test_results_under_32km():
    h = np.array([22100,
                  24000,
                  28800,
                  32000
                  ])  # m
    expected_T = np.array([218.750,
                           220.650,
                           225.450,
                           228.650
                           ])  # K

    expected_rho = np.array([0.062711,
                             0.046267,
                             0.021708,
                             0.013225
                             ])  # kg / m3

    expected_a = np.array([296.50,
                           297.78,
                           301.00,
                           303.13
                           ])  # m/s

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=6)
        assert_almost_equal(a, expected_a[ii], decimal=2)


def test_results_under_47km():
    h = np.array([32200,
                  36000,
                  42000,
                  47000
                  ])  # m
    expected_T = np.array([229.210,
                           239.850,
                           256.650,
                           270.650
                           ])  # K

    expected_rho = np.array([0.012805,
                             0.0070344,
                             0.0028780,
                             0.0014275
                             ])  # kg / m3

    expected_a = np.array([303.50,
                           310.47,
                           321.16,
                           329.80
                           ])  # m / s

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=6)
        assert_almost_equal(a, expected_a[ii], decimal=2)


def test_results_under_51km():
    h = np.array([47200,
                  49000,
                  51000
                  ])  # m
    expected_T = np.array([270.650,
                           270.650,
                           270.650
                           ])  # K

    expected_rho = np.array([0.0013919,
                             0.0011090,
                             0.00086160
                             ])  # kg / m3

    expected_a = np.array([329.80,
                           329.80,
                           329.80
                           ])  # m / s

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=7)
        assert_almost_equal(a, expected_a[ii], decimal=2)


def test_results_under_71km():
    h = np.array([51500,
                  60000,
                  71000
                  ])  # m
    expected_T = np.array([269.250,
                           245.450,
                           214.650
                           ])  # K

    expected_rho = np.array([0.00081298,
                             2.8832e-4,
                             6.4211e-5
                             ])  # kg / m3

    expected_a = np.array([328.94,
                           314.07,
                           293.70
                           ])  # m / s

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=8)
        assert_almost_equal(a, expected_a[ii], decimal=2)


def test_results_under_84km():
    h = np.array([52000,
                  60000,
                  84500
                  ])  # m
    expected_T = np.array([267.850,
                           245.450,
                           187.650
                           ])  # K

    expected_rho = np.array([7.6687e-4,
                             2.8832e-4,
                             7.3914e-6
                             ])  # kg / m3

    expected_a = np.array([328.09,
                           314.07,
                           274.61,
                           ])  # m/s

    for ii, h_ in enumerate(h):
        T, p, rho, a = atmosphere(h_)
        assert_almost_equal(T, expected_T[ii], decimal=3)
        assert_almost_equal(rho, expected_rho[ii], decimal=8)
        assert_almost_equal(a, expected_a[ii], decimal=2)
