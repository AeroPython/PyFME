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

    def update(self, state):
        self._versor = hor2body(self._z_horizon,
                                theta=state.attitude.theta,
                                phi=state.attitude.phi,
                                psi=state.attitude.psi
                                )

        self._vector = self.magnitude * self.versor


class VerticalNewton(Gravity):
    """Vertical gravity model with magnitude varying according to Newton's
    universal law of gravitation.
    """

    def __init__(self):
        self._z_horizon = np.array([0, 0, 1], dtype=float)

    def update(self, state):
        r_squared = (state.position.coord_geocentric @
                     state.position.coord_geocentric)
        self._magnitude = STD_GRAVITATIONAL_PARAMETER / r_squared
        self._versor = hor2body(self._z_horizon,
                                theta=state.attittude.theta,
                                phi=state.attittude.phi,
                                psi=state.attitude.psi
                                )
        self._vector = self.magnitude * self._vector


class LatitudeModel(Gravity):
    # TODO: https://en.wikipedia.org/wiki/Gravity_of_Earth#Latitude_model

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def update(self, system):
        raise NotImplementedError
