"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Angular Velocity
----------------

"""
from abc import abstractmethod

import numpy as np


class AngularVelocity:
    """Angular velocity

    vel_ang : ndarray, shape(3)
        (p [rad/s], q [rad/s], r [rad/s])
    p
    q
    r
    euler_ang_rates : ndarray, shape(3)
        (theta_dot [rad/s], phi_dot [rad/s], psi_dot [rad/s])
    theta
    phi
    psi
    """

    def __init__(self):
        # ANGULAR VELOCITY: (p, q, r)
        self._vel_ang_body = np.zeros(3)  # rad/s
        # EULER ANGLE RATES (theta_dot, phi_dot, psi_dot)
        self._euler_ang_rate = np.zeros(3)  # rad/s

    @abstractmethod
    def update(self, coords, attitude):
        raise NotImplementedError

    @property
    def vel_ang_body(self):
        return self._vel_ang_body

    @property
    def p(self):
        return self._vel_ang_body[0]

    @property
    def q(self):
        return self._vel_ang_body[1]

    @property
    def r(self):
        return self._vel_ang_body[2]

    @property
    def euler_ang_rate(self):
        return self._euler_ang_rate

    @property
    def theta_dot(self):
        return self._euler_ang_rate[0]

    @property
    def phi_dot(self):
        return self._euler_ang_rate[1]

    @property
    def psi_dot(self):
        return self._euler_ang_rate[2]

    @property
    def value(self):
        """Only for testing purposes"""
        return np.hstack((self.vel_ang_body, self.euler_ang_rate))


class BodyAngularVelocity(AngularVelocity):

    def __init__(self, p, q, r, attitude):
        # TODO: docstring
        super().__init__()
        self.update(np.array([p, q, r]), attitude)

    def update(self, coords, attitude):
        self._vel_ang_body[:] = coords
        # TODO: transform angular velocity in body axis to euler angles
        # rates
        self._euler_ang_rate = np.zeros(3)  # rad/s

    def __repr__(self):
        return (f"P: {self.p:.2f} rad/s, "
                f"Q: {self.q:.2f} rad/s, "
                f"R: {self.r:.2f} rad/s")


class EulerAngularRates(AngularVelocity):

    def __init__(self, theta_dot, phi_dot, psi_dot, attitude):
        # TODO: docstring
        super().__init__()
        self.update(np.array([theta_dot, phi_dot, psi_dot]),
                    attitude)

    def update(self, coords, attitude):
        self._euler_ang_rate[:] = coords
        # TODO: transform euler angles rates to angular velocity in body
        #  axis
        self._vel_ang_body[:] = np.zeros(3)  # rad/s
