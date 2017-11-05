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
    def set_angular_accel(self, coords, attitude):

        if acc_ang_body is not None and euler_ang_acc is not None:
            raise ValueError("Only values for acc_ang_body or euler_ang_acc"
                             " can be given")
        elif acc_ang_body is not None:
            self._acc_ang_body[:] = acc_ang_body
            # TODO: transform angular acc in body axis to euler angles
            # acc
            self._euler_ang_acc = np.zeros(3)  # rad/s
        elif euler_ang_acc is not None:
            self._euler_ang_acc[:] = euler_ang_acc
            # TODO: transform euler angles acc to angular acceleration in body
            #  axis
            self._acc_ang_body[:] = np.zeros(3)  # rad/s
        else:
            raise ValueError("acc_ang_body or euler_angles must be given")

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


class BodyAngularAcceleration(AngularAcceleration):

    def __init__(self, p_dot, q_dot, r_dot, attitude):
        super().__init__()
        self.set_angular_accel(np.array([p_dot, q_dot, r_dot]), attitude)

    def set_angular_accel(self, coords, attitude):
        self._acc_ang_body[:] = coords
        # TODO: transform angular acc in body axis to euler angles
        # acc
        self._euler_ang_acc = np.zeros(3)  # rad/s


class EulerAngularAcceleration(AngularAcceleration):

    def __init__(self, theta_dot, phi_dot, psi_dot, attitude):
        super().__init__()
        self.set_angular_accel(np.array([theta_dot, phi_dot, psi_dot]),
                               attitude)

    def set_angular_accel(self, coords, attitude):
        self._euler_ang_acc[:] = coords
        # TODO: transform euler angles acc to angular acceleration in body
        #  axis
        self._acc_ang_body[:] = np.zeros(3)  # rad/s