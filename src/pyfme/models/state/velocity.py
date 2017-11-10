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
from pyfme.utils.coordinates import body2hor, hor2body


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
        self._vel_NED = body2hor(value,
                                 attitude.theta,
                                 attitude.phi,
                                 attitude.psi)  # m/s

    def __repr__(self):
        return f"u: {self.u} m/s, v: {self.v} m/s, w: {self.w} m/s"


class NEDVelocity(Velocity):
    def __init__(self, vn, ve, vd, attitude):
        # TODO: docstring
        super().__init__()
        self.update(np.array([vn, ve, vd]), attitude)

    def update(self, value, attitude):
        self._vel_NED[:] = value
        self._vel_body = hor2body(value,
                                  attitude.theta,
                                  attitude.phi,
                                  attitude.psi)  # m/s

    def __repr__(self):
        return (f"V_north: {self.v_north} m/s,"
                f"V_east: {self.v_east} m/s, "
                f"V_down: {self.v_down} m/s")
