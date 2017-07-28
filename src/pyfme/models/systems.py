"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic Systems
---------------

"""
from abc import abstractmethod, abstractstaticmethod

import numpy as np
from scipy.integrate import ode


class System(object):

    def __init__(self, dynamic_system):

        # Dynamic system
        self._dynamic_system = dynamic_system

        # POSITION
        # Geodetic coordinates: (geodetic lat, lon, height above ellipsoid)
        self.geodetic_coordinates = np.zeros(3)  # rad
        # Geocentric coordinates (rotating with Earth): (x_geo, y_geo, z_geo)
        self.geocentric_coordinates = np.zeros(3)  # m
        # Earth coordinates (x_earth, y_earth, z_earth)
        self.earth_coordinates = np.zeros(3)  # m

        # ATTITUDE
        # Euler angles (psi, theta, phi)
        self.euler_angles = np.zeros(3)  # rad
        # Quaternions (q0, q1, q2, q3)
        self.quaternions = np.zeros(4)

        # ABSOLUTE VELOCITY.
        # Body axis
        self.vel_body = np.zeros(3)  # m/s
        # Local horizon (NED)
        self.vel_NED = np.zeros(3)  # m/s

        # ANGULAR VELOCITY: (p, q, r)
        self.vel_ang = np.zeros(3)  # rad/s

        # ABSOLUTE ACCELERATION
        # Body axis
        self.accel_body = np.zeros(3)  # m/s²
        # Local horizon (NED)
        self.accel_NED = np.zeros(3)  # m/s²

        # ANGULAR ACCELERATION
        self.accel_ang = np.zeros(3)  # rad/s²

    @property
    def lat(self):
        return self.geodetic_coordinates[0]

    @property
    def lon(self):
        return self.geodetic_coordinates[1]

    @property
    def height(self):
        return self.geodetic_coordinates[2]

    @property
    def x_geo(self):
        return self.geocentric_coordinates[0]

    @property
    def y_geo(self):
        return self.geocentric_coordinates[1]

    @property
    def z_geo(self):
        return self.geocentric_coordinates[2]

    @property
    def x_earth(self):
        return self.earth_coordinates[0]

    @property
    def y_earth(self):
        return self.earth_coordinates[1]

    @property
    def z_earth(self):
        return self.earth_coordinates[2]

    @property
    def psi(self):
        return self.euler_angles[0]

    @property
    def theta(self):
        return self.euler_angles[1]

    @property
    def phi(self):
        return self.euler_angles[2]

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
    def v_north(self):
        return self.vel_NED[0]

    @property
    def v_east(self):
        return self.vel_NED[1]

    @property
    def v_down(self):
        return self.vel_NED[2]

    @property
    def p(self):
        return self.vel_ang[0]

    @property
    def q(self):
        return self.vel_ang[1]

    @property
    def r(self):
        return self.vel_ang[2]

    @property
    def time(self):
        return self._dynamic_system.time


class DynamicSystem(object):

    def __init__(self, state, use_jacobian=None, integrator=None,
                 callback=None, **integrator_params):

        self.state = state

        self._equations = self.dynamic_system_equations

        if use_jacobian:
            self._jacobian = self.dynamic_system_jacobian
        else:
            self._jacobian = None

        self._ode = ode(self._equations, self._jacobian)

        if integrator is None:
            integrator = 'dopri5'

        self._ode.set_integrator(integrator, **integrator_params)
        self._ode.set_initial_value(self.state)

        if callback:
            self._ode.set_solout(callback)

    @property
    def time(self):
        return self._ode.t

    def propagate(self, dt, mass, inertia, forces, moments):

        t = self._ode.t + dt

        self._ode.set_f_params(mass, inertia, forces, moments)

        if self._ode.jac:
            self._ode.set_jac_params(mass, inertia, forces, moments)

        self.state = self._ode.integrate(t)

        if self._ode.successful():
            return self.state

    @abstractmethod
    def dynamic_system_state_to_full_system_state(self):
        raise NotImplementedError

    @abstractstaticmethod
    def dynamic_system_equations(time, state_vector, mass, inertia, forces,
                                 moments):
        raise NotImplementedError

    @abstractstaticmethod
    def dynamic_system_jacobian(state_vector, mass, inertia, forces, moments):
        raise NotImplementedError


