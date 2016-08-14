# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Atmosphere
----------

"""

from math import exp, sqrt
from abc import abstractmethod

from pyfme.models.constants import GAMMA_AIR, R_AIR, GRAVITY
from pyfme.utils.altimetry import geometric2geopotential


class Atmosphere(object):

    def __init__(self):
        # TODO: doc
        self._gamma = None  # Adiabatic index or ratio of specific heats
        self._R_g = None  # Gas constant  J/(Kg·K)
        self._g0 = None  # Gravity  m/s^2

        self.geopotential_alt = None  # Current geopotential height (m).
        self.pressure_altitude = None  # Current pressure altitude (m).
        self.T = None  # Temperature (K).
        self.p = None  # Pressure (atm).
        self.rho = None  # Density (kg/m³).
        self.a = None  # Speed of sound (m/s).

    def update(self, system):
        """Update atmosphere state for the given system state.

        Parameters
        ----------
        system : System object
            System object with attribute alt_geop (geopotential
            altitude.

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
         # Geopotential altitude
        self.geopotential_alt = geometric2geopotential(system.height)
        self.T, self.p, self.rho, self.a = self.__call__(self.geopotential_alt)

    @abstractmethod
    def __call__(self, h):
        pass


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
        super().__init__()
        self._gamma = GAMMA_AIR  # Adiabatic index or ratio of specific heats
        self._R_g = R_AIR  # Gas constant  J/(Kg·K)
        self._g0 = GRAVITY  # Gravity  m/s^2
        # Layer constants
        self._h0 = (0, 11000, 20000, 32000, 47000, 51000, 71000, 84500)  # m
        self._T0_layers = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 
                           214.65)  # K
        self._p0_layers = (101325.0, 22632.1, 5474.89, 868.019, 110.906, 
                           66.9389, 3.95642)  # Pa
        self._alpha_layers = (-0.0065, 0, 0.001, 0.0028, 0, -0.0028,
                              -0.002)  # K / m

        # Initialize at h=0
        self.h = 0  # Current height (m).
        self.T = self._T0_layers[0]  # Temperature (K).
        self.p = self._p0_layers[0]  # Pressure (atm).
        self.rho = self.p / (self._R_g * self.T)
        self.a = sqrt(self._gamma * self._R_g * self.T)

    def __call__(self, h):
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
        Note that this method will calculate the atmosphere `T, p, rho,
        a`  values corresponding to the given geopotential altitude, but the
        atmosphere object will not be updated. Use update instead to
        update the atmosphere.

        Check layers and reference values in [2].

        References
        ----------
        .. [1] U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
            Washington, D.C., 1976
        .. [2] https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere

        """
        g0 = self._g0
        R_a = self._R_g
        gamma = self._gamma

        if h < 0.0:
            raise ValueError("Altitude cannot be less than 0 m.")

        elif self._h0[0] <= h < self._h0[1]:  # Troposphere
            T0 = self._T0_layers[0]
            p0 = self._p0_layers[0]
            alpha = self._alpha_layers[0]

            T = T0 + alpha * h
            p = p0 * (T0 / (T0 + alpha * h)) ** (g0 / (R_a * alpha))

        elif self._h0[1] <= h < self._h0[2]:  # Tropopause
            T0 = self._T0_layers[1]
            p0 = self._p0_layers[1]
            # alpha = self._alpha_layers[1]
            h_diff = h - self._h0[1]

            T = T0
            p = p0 * exp(-g0 * h_diff / (R_a * T0))

        elif self._h0[2] <= h < self._h0[3]:  # Stratosphere 1
            T0 = self._T0_layers[2]
            p0 = self._p0_layers[2]
            alpha = self._alpha_layers[2]
            h_diff = h - self._h0[2]

            T = T0 + alpha * h_diff
            p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        elif self._h0[3] <= h < self._h0[4]:  # Stratosphere 2
            T0 = self._T0_layers[3]
            p0 = self._p0_layers[3]
            alpha = self._alpha_layers[3]
            h_diff = h - self._h0[3]

            T = T0 + alpha * h_diff
            p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        elif self._h0[4] <= h < self._h0[5]:  # Stratopause
            T0 = self._T0_layers[4]
            p0 = self._p0_layers[4]
            # alpha = self._alpha_layers[4]
            h_diff = h - self._h0[4]

            T = T0
            p = p0 * exp(-g0 * h_diff / (R_a * T0))

        elif self._h0[5] <= h < self._h0[6]:  # Mesosphere 1
            T0 = self._T0_layers[5]
            p0 = self._p0_layers[5]
            alpha = self._alpha_layers[5]
            h_diff = h - self._h0[5]

            T = T0 + alpha * h_diff
            p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        elif self._h0[6] <= h <= self._h0[7]:  # Mesosphere 2
            T0 = self._T0_layers[6]
            p0 = self._p0_layers[6]
            alpha = self._alpha_layers[6]
            h_diff = h - self._h0[6]

            T = T0 + alpha * h_diff
            p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

        else:
            raise ValueError(
                "Altitude cannot be greater than {} m.".format(self._h0[7]))
        rho = p / (R_a * T)
        a = sqrt(gamma * R_a * T)
        return T, p, rho, a
