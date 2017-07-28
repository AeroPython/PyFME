"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Gravity Models
--------------

"""
from abc import abstractmethod

import numpy as np

from pyfme.models.constants import GRAVITY, STD_GRAVITATIONAL_PARAMETER
from pyfme.utils.coordinates import hor2body


class Gravity(object):
    """Generic gravity model"""

    def __init__(self):
        self._magnitude = None
        self._versor = np.zeros([3])  # Body axis
        self._vector = np.zeros([3])  # Body axis

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def versor(self):
        return self._versor

    @property
    def vector(self):
        return self._vector

    @abstractmethod
    def update(self, system):
        pass


class VerticalConstant(Gravity):
    """Vertical constant gravity model.
    """

    def __init__(self):
        self._magnitude = GRAVITY
        self._z_horizon = np.array([0, 0, 1], dtype=float)

    def update(self, system):
        self._versor = hor2body(self._z_horizon,
                                theta=system.theta,
                                phi=system.phi,
                                psi=system.psi
                                )

        self._vector = self.magnitude * self.versor


class VerticalNewton(Gravity):
    """Vertical gravity model with magnitude varying according to Newton's
    universal law of gravitation.
    """

    def __init__(self):
        self._z_horizon = np.array([0, 0, 1], dtype=float)

    def update(self, system):
        r_squared = system.coord_geocentric @ system.coord_geocentric
        self._magnitude = STD_GRAVITATIONAL_PARAMETER / r_squared
        self._versor = hor2body(self._z_horizon,
                                theta=system.theta,
                                phi=system.phi,
                                psi=system.psi
                                )
        self._vector = self.magnitude * self._vector


class LatitudeModel(Gravity):
    # TODO: https://en.wikipedia.org/wiki/Gravity_of_Earth#Latitude_model

    def __init__(self):
        raise NotImplementedError
