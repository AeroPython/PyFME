"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic Systems
---------------

"""
from abc import abstractmethod

import numpy as np
from scipy.integrate import ode

from pyfme.utils.coordinates import body2hor


class System(object):
    """Generic system class contains the state vector and other derived
    variables related to the system's state.
    """

    def __init__(self, lat, lon, h, psi=0, x_earth=0, y_earth=0):

        # ABOUT UNITS: assume international system units unless otherwise
        # indicated. (meters, kilograms, seconds, kelvin, Newtons, Pascal,
        # radians...)
        # POSITION
        # Geographic coordinates: (geodetic lat, lon, height above ellipsoid)
        self.coord_geographic = np.array([lat, lon, h], dtype=float)
        # Geocentric coordinates (rotating with Earth): (x_geo, y_geo, z_geo)
        # TODO: implement geographic2geocentric conversion:
        # self.coord_geocentric = np.zeros_like([3], dtype=float)

        # Earth coordinates (Axis parallel to local horizon NED at h=0 at
        # the initial position of the airplane): (x_earth, y_earth, z_earth)
        self.coord_earth = np.array([x_earth, y_earth, -h], dtype=float)

        # ATTITUDE (psi, theta, phi).
        self.euler_angles = np.zeros([3], dtype=float)
        self.euler_angles[0] = psi
        # TODO: convert to quaternions
        # self.quaternions = np.zeros([4], dtype=float)

        # ABSOLUTE VELOCITY.
        # Body:
        self.vel_body = np.zeros_like([3], dtype=float)
        # Local horizon (NED):
        self.vel_NED = np.zeros_like([3], dtype=float)

        # ANGULAR VELOCITY: (p, q, r)
        self.vel_ang = np.zeros_like([3], dtype=float)

        # Last time step dt
        self.dt = None

    # TODO: guarantee that if euler angles change <--> quaternions change
    # TODO: guarantee that if geographic change <-->  geocentric change
    # TODO: guarantee that if body vels change <-->  horizon vels change
    @property
    def lat(self):
        return self.coord_geographic[0]

    @property
    def lon(self):
        return self.coord_geographic[1]

    @property
    def height(self):
        return self.coord_geographic[2]

    @property
    def x_geo(self):
        return self.coord_geocentric[0]

    @property
    def y_geo(self):
        return self.coord_geocentric[1]

    @property
    def z_geo(self):
        return self.coord_geocentric[2]

    @property
    def x_earth(self):
        return self.coord_earth[0]

    @property
    def y_earth(self):
        return self.coord_earth[1]

    @property
    def z_earth(self):
        return self.coord_earth[2]

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

    @abstractmethod
    def _propagate_state_vector(self, aircraft, dt):
        pass

    @abstractmethod
    def set_initial_state_vector(self):
        pass

    def propagate(self, aircraft, environment, dt=0.01):
        pass


class EulerFlatEarth(System):
    """Euler flat Earth equations system"""

    def __init__(self, lat, lon, h, psi=0, x_earth=0, y_earth=0,
                 integrator='dopri5', use_jac=False, **integrator_params):
        """Initialize the equations of the chosen model and selects the
        integrator. Check `scipy.integrate.ode` to see available integrators.

        If use_jac = True the jacobian of the equations is used for the
        integration.

        Parameters
        ----------
        lat, lon, h: float
            Latitude, longitude and height (rad, rad, m).
        psi, x_earth, y_earth: float, opt
            Yaw angle and initial Earth position (rad, m, m).
        integrator: str, optional
            Any allowed integrator for `ode` class: "vode", "zvode", "lsoda",
            "dopri5", "dop853".
        use_jac : bool, optional
            Use analytical jacobians of system equations.
        """
        super().__init__(lat, lon, h, psi, x_earth, y_earth)
        # State vector must be initialized with set_initial_state_vector() method
        self.state_vector = None

        from pyfme.models.euler_flat_earth import lamceq, kaeq, kleq
        self.lamceq = lamceq

        if use_jac:
            from pyfme.models.euler_flat_earth import lamceq_jac, kaeq_jac
            jac_LM_and_AM = lamceq_jac
            jac_att = kaeq_jac
            jac_nav = None  # not implemented
        else:
            jac_LM_and_AM = None
            jac_att = None
            jac_nav = None

        self._ode_lamceq = ode(lamceq, jac=jac_LM_and_AM)
        self._ode_kaqeq = ode(kaeq, jac=jac_att)
        self._ode_kleq = ode(kleq, jac=jac_nav)

        self._ode_lamceq.set_integrator(integrator, **integrator_params)
        self._ode_kaqeq.set_integrator(integrator, **integrator_params)
        self._ode_kleq.set_integrator(integrator, **integrator_params)

    @property
    def height(self):
        return -self.coord_earth[2]

    def set_initial_state_vector(self):
        """
        Set the initial values of the state vector for system integration
        once the values for the involved variables have been assigned or the
        system has been trimmed.
        """

        self.vel_NED = body2hor(self.vel_body, theta=self.theta,
                                phi=self.phi, psi=self.psi)
        self.state_vector = np.array([
            self.u, self.v, self.w,
            self.p, self.q, self.r,
            self.theta, self.phi, self.psi,
            self.x_earth, self.y_earth, self.z_earth
        ])

        self._ode_lamceq.set_initial_value(y=self.state_vector[0:6])
        self._ode_kaqeq.set_initial_value(y=self.state_vector[6:9])
        self._ode_kleq.set_initial_value(y=self.state_vector[9:12])

    def _propagate_state_vector(self, aircraft, dt):
        """
        Performs integration step for actual_time + dt and returns the state
        vector
        """
        mass = aircraft.mass
        inertia = aircraft.inertia
        forces = aircraft.total_forces
        moments = aircraft.total_moments

        t = self._ode_lamceq.t + dt

        self._ode_lamceq.set_f_params(mass, inertia, forces, moments)
        velocities = self._ode_lamceq.integrate(t)

        if self._ode_lamceq.successful():
            self._ode_kaqeq.set_f_params(velocities[3:])
            attitude_angles = self._ode_kaqeq.integrate(t)
        else:
            raise RuntimeError('Integration of Linear and angular momentum \
                                equations was not successful')

        if self._ode_kaqeq.successful():
            self._ode_kleq.set_f_params(velocities[0:3], attitude_angles)
            position = self._ode_kleq.integrate(t)
        else:
            raise RuntimeError('Integration of attitude equations was not \
                                successful')

        if self._ode_kleq.successful():
            self.state_vector[0:6] = velocities[:]
            self.state_vector[6:9] = attitude_angles[:]
            self.state_vector[9:12] = position[:]
        else:
            raise RuntimeError('Integration of navigation equations was not \
                                successful')

        return self.state_vector

    def propagate(self, aircraft, dt=0.01):
        """Propagate the state vector and update the rest of variables.

        Parameters
        ----------
        aircraft : Aircraft
            Aircraft model for simulation is used to get forces.
        """
        self.dt = dt
        self._propagate_state_vector(aircraft, dt)

        # TODO: update the rest of variables.
        self.vel_body = self.state_vector[0:3]
        self.vel_ang = self.state_vector[3:6]
        self.euler_angles[0] = self.state_vector[8]  # psi
        self.euler_angles[1] = self.state_vector[6]  # theta
        self.euler_angles[2] = self.state_vector[7]  # phi
        self.coord_earth = self.state_vector[9:12]

        # Set psi between 0 and 2*pi
        # FIXME: check the conversion to keep angle between 0 and 2pi again
        self.euler_angles[0] = np.arctan2(np.sin(self.psi), np.cos(
            self.psi)) % (2*np.pi)
        self.euler_angles[2] = np.arctan2(np.sin(self.phi), np.cos(self.phi))

        self.vel_NED = body2hor(self.vel_body, theta=self.theta, phi=self.phi,
                                psi=self.psi)
