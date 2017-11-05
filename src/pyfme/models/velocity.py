"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Velocity
--------

The aircraft state has
"""
from abc import abstractmethod

import numpy as np


# TODO: think about generic changes from body to horizon that could be used for
# velocity, accelerations...
# If also changes from attitude of elements in the body (such as sensors) to
# body and horizon coordinates are implemented it would be useful!
class Velocity:
    """Velocity

    Attributes
    ----------

    vel_body : ndarray, shape(3)
        (u [m/s], v [m/s], w [m/s])
    u
    v
    w
    vel_NED : ndarray, shape(3)
        (v_north [m/s], v_east [m/s], v_down [m/s])
    v_north
    v_east
    v_down
    """

    def __init__(self):
        # Body axis
        self._vel_body = np.zeros(3)  # m/s
        # Local horizon (NED)
        self._vel_NED = np.zeros(3)  # m/s

    @abstractmethod
    def update(self, coords, attitude):
        raise NotImplementedError

    @property
    def vel_body(self):
        return self._vel_body

    @property
    def u(self):
        return self.vel_body[0]

    @property
    def v(self):
        return self.vel_body[1]

    @property
    def w(self):
        return self.vel_body[2]

    @property
    def vel_NED(self):
        return self._vel_NED

    @property
    def v_north(self):
        return self._vel_NED[0]

    @property
    def v_east(self):
        return self._vel_NED[1]

    @property
    def v_down(self):
        return self._vel_NED[2]


class BodyVelocity(Velocity):

    def __init__(self, u, v, w, attitude):
        # TODO: docstring
        super().__init__()
        self.update(np.array([u, v, w]), attitude)

    def update(self, value, attitude):
        self._vel_body[:] = value
        # TODO: transform body vel to horizon vel using attitude
        self._vel_NED = np.zeros(3)  # m/s


class NEDVelocity(Velocity):
    def __init__(self, vn, ve, vd, attitude):
        # TODO: docstring
        super().__init__()
        self.update(np.array([vn, ve, vd]), attitude)

    def update(self, value, attitude):
        self._vel_NED[:] = value
        # TODO: transform horizon vel to body vel using attitude
        self._vel_body = np.zeros(3)  # m/s