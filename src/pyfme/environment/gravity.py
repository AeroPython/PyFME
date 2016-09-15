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
        self.magnitude = None
        self.unitary_vector = np.zeros([3])  # Body axis
        self.vector = np.zeros([3])  # Body axis

    @abstractmethod
    def update(self, system):
        pass


class VerticalConstant(Gravity):
    """Vertical constant gravity model.
    """

    def __init__(self):
        Gravity.__init__(self)
        self.magnitude = GRAVITY
        self._z_horizon = np.array([0, 0, 1], dtype=float)

    def update(self, system):
        self.unitary_vector = hor2body(self._z_horizon,
                                       theta=system.theta,
                                       phi=system.phi,
                                       psi=system.psi)
        self.vector = self.magnitude * self.unitary_vector


class VerticalNewton(Gravity):
    """Vertical gravity model with magnitude varying according to Newton's
    universal law of gravitation.
    """

    def __init__(self):
        Gravity.__init__(self)
        self._z_horizon = np.array([0, 0, 1], dtype=float)

    def update(self, system):
        r_squared = system.coord_geocentric @ system.coord_geocentric
        self.magnitude = STD_GRAVITATIONAL_PARAMETER / r_squared
        self.unitary_vector = hor2body(self._z_horizon,
                                       theta=system.theta,
                                       phi=system.phi,
                                       psi=system.psi)
        self.vector = self.magnitude * self.unitary_vector


class LatitudeModel(Gravity):
    # TODO: https://en.wikipedia.org/wiki/Gravity_of_Earth#Latitude_model

    def __init__(self):
        raise NotImplementedError
