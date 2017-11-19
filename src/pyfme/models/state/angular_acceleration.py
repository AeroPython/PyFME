"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Angular Acceleration
--------------------

"""
from abc import abstractmethod

import numpy as np


class AngularAcceleration:
    """Angular Accelerations

    Attributes
    ----------
    accel_ang : ndarray, shape(3)
        (p_dot [rad/s²], q_dot [rad/s²], r_dot [rad/s²])
    p_dot
    q_dot
    r_dot
    euler_ang_acc : ndarray, shape(3)
        (theta_2dot [rad/s²], phi_2dot [rad/s²], psi_2dot [rad/s²])
    theta_2dot
    phi_2dot
    psi_2dot
    """

    def __init__(self):
        # ANGULAR VELOCITY: (p_dot, q_dot, r_dot)
        self._acc_ang_body = np.zeros(3)  # rad/s
        # EULER ANGLE RATES (theta_dot2, phi_dot2, psi_dot2)
        self._euler_ang_acc = np.zeros(3)  # rad/s

    @abstractmethod
    def update(self, coords, attitude):
        raise ValueError

    @property
    def acc_ang_body(self):
        return self._acc_ang_body

    @property
    def p_dot(self):
        return self._acc_ang_body[0]

    @property
    def q_dot(self):
        return self._acc_ang_body[1]

    @property
    def r_dot(self):
        return self._acc_ang_body[2]

    @property
    def euler_ang_acc(self):
        return self._euler_ang_acc

    @property
    def theta_2dot(self):
        return self._euler_ang_acc[0]

    @property
    def phi_2dot(self):
        return self._euler_ang_acc[1]

    @property
    def psi_2dot(self):
        return self._euler_ang_acc[2]

    @property
    def value(self):
        """Only for testing purposes"""
        return np.hstack((self.acc_ang_body, self.euler_ang_acc))


class BodyAngularAcceleration(AngularAcceleration):

    def __init__(self, p_dot, q_dot, r_dot, attitude):
        super().__init__()
        self.update(np.array([p_dot, q_dot, r_dot]), attitude)

    def update(self, coords, attitude):
        self._acc_ang_body[:] = coords
        # TODO: transform angular acc in body axis to euler angles
        # acc
        self._euler_ang_acc = np.zeros(3)  # rad/s

    def __repr__(self):
        rv = (f"P_dot: {self.p_dot:.2f} rad/s², "
              f"Q_dot: {self.q_dot:.2f} rad/s², "
              f"R_dot: {self.r_dot:.2f} rad/s²")
        return rv


class EulerAngularAcceleration(AngularAcceleration):

    def __init__(self, theta_dot, phi_dot, psi_dot, attitude):
        super().__init__()
        self.update(np.array([theta_dot, phi_dot, psi_dot]),
                    attitude)

    def update(self, coords, attitude):
        self._euler_ang_acc[:] = coords
        # TODO: transform euler angles acc to angular acceleration in body
        #  axis
        self._acc_ang_body[:] = np.zeros(3)  # rad/s
