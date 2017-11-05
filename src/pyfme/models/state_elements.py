"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

State elements
--------------

The aircraft state has always the same elements even if they are expressed
in a different way. For example, attitude can be expressed with Euler angles
or quaternions, position with geodetic coordinates or Earth coordinates...

This module provides class to represent:
  * position
  * attitude
  * velocity
  * angular velocity
  * acceleration
  * angular acceleration

"""
from abc import abstractmethod

import numpy as np


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
    def set_velocity(self, coords, attitude):
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
        self.set_velocity(np.array([u, v, w]), attitude)

    def set_velocity(self, value, attitude):
        self._vel_body[:] = value
        # TODO: transform body vel to horizon vel using attitude
        self._vel_NED = np.zeros(3)  # m/s


class NEDVelocity(Velocity):
    def __init__(self, vn, ve, vd, attitude):
        # TODO: docstring
        super().__init__()
        self.set_velocity(np.array([vn, ve, vd]), attitude)

    def set_velocity(self, value, attitude):
        self._vel_NED[:] = value
        # TODO: transform horizon vel to body vel using attitude
        self._vel_body = np.zeros(3)  # m/s


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

    def set_angular_velocity(self, attitude, vel_ang_body=None,
                             euler_ang_rates=None):

        if vel_ang_body is not None and euler_ang_rates is not None:
            raise ValueError("Only values for vel_ang_body or euler_ang_rates"
                             " can be given")
        elif vel_ang_body is not None:
            self._vel_ang_body[:] = vel_ang_body
            # TODO: transform angular velocity in body axis to euler angles
            # rates
            self._euler_ang_rate = np.zeros(3)  # rad/s
        elif euler_ang_rates is not None:
            self._euler_ang_rate[:] = euler_ang_rates
            # TODO: transform euler angles rates to angular velocity in body
            #  axis
            self._vel_ang_body[:] = np.zeros(3)  # rad/s
        else:
            raise ValueError("vel_ang_body or euler_angles must be given")

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

    def set_acceleration(self, attitude, accel_body=None, accel_NED=None):
        if accel_body is not None and accel_NED is not None:
            raise ValueError("Only values for accel_body or accel_NED can be "
                             "given")
        elif accel_NED is None:
            self._accel_body[:] = accel_body
            # TODO: transform body vel to horizon vel using attitude
            self._accel_NED = np.zeros(3)  # m/s
        elif accel_body is None:
            self._accel_NED[:] = accel_NED
            # TODO: transform horizon vel to body vel using attitude
            self._accel_body = np.zeros(3)  # m/s
        else:
            raise ValueError("accel_body or accel_NED must be given")

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

    def set_angular_velocity(self, attitude, acc_ang_body=None,
                             euler_ang_acc=None):

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
