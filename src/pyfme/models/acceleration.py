"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Acceleration
------------

"""
from abc import abstractmethod

import numpy as np


class Acceleration:
    """Acceleration

    Attributes
    ----------
    accel_body : ndarray, shape(3)
        (u_dot [m/s²], v_dot [m/s²], w_dot [m/s²])
    u_dot
    v_dot
    w_dot
    accel_NED : ndarray, shape(3)
        (VN_dot [m/s²], VE_dot [m/s²], VD_dot [m/s²])
    VN_dot
    VE_dot
    VD_dot
    """

    def __init__(self):
        # Body axis
        self._accel_body = np.zeros(3)  # m/s²
        # Local horizon (NED)
        self._accel_NED = np.zeros(3)  # m/s²

    @abstractmethod
    def set_acceleration(self, coords, attitude):
        raise NotImplementedError

    @property
    def accel_body(self):
        return self._accel_body

    @property
    def u_dot(self):
        return self._accel_body[0]

    @property
    def v_dot(self):
        return self._accel_body[1]

    @property
    def w_dot(self):
        return self._accel_body[2]

    @property
    def accel_NED(self):
        return self._accel_NED

    @property
    def v_north_dot(self):
        return self._accel_NED[0]

    @property
    def v_east_dot(self):
        return self._accel_NED[1]

    @property
    def v_down_dot(self):
        return self._accel_NED[2]


class BodyAcceleration(Acceleration):

    def __init__(self, u_dot, v_dot, w_dot, attitude):
        super().__init__()
        self.set_acceleration(np.array([u_dot, v_dot, w_dot]), attitude)

    def set_acceleration(self, coords, attitude):
        self._accel_body[:] = coords
        # TODO: transform body vel to horizon vel using attitude
        self._accel_NED = np.zeros(3)  # m/s


class NEDAcceleration(Acceleration):

    def __init__(self, vn_dot, ve_dot, vd_dot, attitude):
        super().__init__()
        self.set_acceleration(np.array([vn_dot, ve_dot, vd_dot]), attitude)

    def set_acceleration(self, coords, attitude):
        self._accel_NED[:] = coords
        # TODO: transform horizon vel to body vel using attitude
        self._accel_body = np.zeros(3)  # m/s