# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Atmosphere
----------

"""

from math import exp, sqrt

from pyfme.models.constants import GAMMA_AIR, R_AIR, GRAVITY

class Atmosphere(object):

    def __init__(self):
        # TODO: doc
        self._gamma = GAMMA_AIR
        self._R_a = R_AIR
        self._g0 = GRAVITY

        self.h = None  # Current height (m).
        self.T = None  # Temperature (K).
        self.p = None  # Pressure (atm).
        self.rho = None  # Density (kg/m³).
        self.a = None  # Speed of sound (m/s).

# FIXME: TESTS to be modified
class ISA1976(Atmosphere):
    """
    International Standard Atmosphere 1976
    --------------------------------------
    Implementation based on:
    .. [1] U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
            Washington, D.C., 1976

    From: https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere

    =========  ============  =========  ===========  =============
    Subscript  Geopotential  Static     Standard     Temperature
               altitude      Pressure   Temperature  Lapse Rate
               above MSL     (pascals)  (K)          (K/m)
               (m)
    =========  ============  =========  ===========  =============
    0          0             101325     288.15       -0.0065
    1          11000         22632.1    216.65        0
    2          20000         5474.89    216.65        0.001
    3          32000         868.019    228.65        0.0028
    4          47000         110.906    270.65        0
    5          51000         66.9389    270.65       -0.0028
    6          71000         3.95642    214.65       -0.002
    =========  ============  =========  ===========  =============
    """

    def __init__(self):
        Atmosphere.__init__(self)
        # Layer constants
        self._h0 = (0, 11000, 20000, 32000, 47000, 51000, 71000, 84500)  # m
        self._T0_layers = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 
                           214.65)  # K
        self._p0_layers = (101325.0, 22632.1, 5474.89, 868.019, 110.906, 
                           66.9389, 3.95642)  # Pa
        self._alpha_layers = (-0.0065, 0, 0.001, 0.0028, 0, -0.0028,
                              -0.002)  # K / m

    def __call__(self, h):
        # Fixme: check documentation & class documentation
        """ISA 1976 Standard atmosphere temperature, pressure and density.

        Parameters
        ----------
        h : float
            Geopotential altitude (m). h values must range from 0 to 84500 m.

        Returns
        -------
        T : float
            Temperature (K).
        p : float
            Pressure (Pa).
        rho : float
            Density (kg/m³)
        a : float
            Sound speed at flight level (m/s)

        Raises
        ------
        ValueError
            If the value of the altitude is outside the defined layers.

        Notes
        -----
        Check layers and reference values in [2].

        References
        ----------
        .. [1] U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
            Washington, D.C., 1976
        .. [2] https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere

        """

        g0 = self._g0
        R_a = self.R_a
        gamma = self._gamma

        if h < 0.0:
            raise ValueError("Altitude cannot be less than 0 m.")

        elif self._h0[0] <= h < self._h0[1]:  # Troposphere
            T0 = self._T0_layers[0]
            p0 = self._p0_layers[0]
            alpha = self._alpha_layers[0]

            self.T = T0 + alpha * h
            self.p = p0 * (T0 / (T0 + alpha * h)) ** (g0 / (R_a * alpha))

        elif self._h0[1] <= h < self._h0[2]:  # Tropopause
            T0 = self._T0_layers[1]
            p0 = self._p0_layers[1]
            # alpha = self._alpha_layers[1]
            h_diff = h - self._h0[1]

            self.T = T0
            self.p = p0 * exp(-g0 * h_diff / (R_a * T0))

        elif self._h0[2] <= h < self._h0[3]:  # Stratosphere 1
            T0 = self._T0_layers[2]
            p0 = self._p0_layers[2]
            alpha = self._alpha_layers[2]
            h_diff = h - self._h0[2]

            self.T = T0 + alpha * h_diff
            self.p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        elif self._h0[3] <= h < self._h0[4]:  # Stratosphere 2
            T0 = self._T0_layers[3]
            p0 = self._p0_layers[3]
            alpha = self._alpha_layers[3]
            h_diff = h - self._h0[3]

            self.T = T0 + alpha * h_diff
            self.p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        elif self._h0[4] <= h < self._h0[5]:  # Stratopause
            T0 = self._T0_layers[4]
            p0 = self._p0_layers[4]
            # alpha = self._alpha_layers[4]
            h_diff = h - self._h0[4]

            self.T = T0
            self.p = p0 * exp(-g0 * h_diff / (R_a * T0))

        elif self._h0[5] <= h < self._h0[6]:  # Mesosphere 1
            T0 = self._T0_layers[5]
            p0 = self._p0_layers[5]
            alpha = self._alpha_layers[5]
            h_diff = h - self._h0[5]

            self.T = T0 + alpha * h_diff
            self.p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        elif self._h0[6] <= h <= self._h0[7]:  # Mesosphere 2
            T0 = self._T0_layers[6]
            p0 = self._p0_layers[6]
            alpha = self._alpha_layers[6]
            h_diff = h - self._h0[6]

            self.T = T0 + alpha * h_diff
            self.p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        else:
            raise ValueError(
                "Altitude cannot be greater than {} m.".format(self._h0[7]))
        self.h = h
        self.rho = self.p / (R_a * self.T)
        self.a = sqrt(gamma * R_a * self.T)
        return self.T, self.p, self.rho, self.a

