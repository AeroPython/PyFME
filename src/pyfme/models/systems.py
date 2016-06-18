"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic Systems
---------------

"""

import numpy as np
from scipy.integrate import ode

class System(object):

    def __init__(self):

        # ABOUT UNITS: assume international system units unless otherwise
        # indicated. (meters, kilograms, seconds, kelvin, Newtons, Pascal,
        # radians...)
        # POSITION
        # Geographic coordinates: (geodetic lat, lon, height above ellipsoid)
        self.coord_geographic = np.zeros([3], dtype=float)
        # TODO: convert into properties.
        # Geocentric coordinates (rotating with Earth): (x_geo, y_geo, z_geo)
        self.coord_geocentric = np.zeros_like([3], dtype=float)
        # Earth coordinates (Axis parallel to local horizon NED at h=0 at
        # the initial position of the airplane): (x_earth, y_earth, z_earth)
        self.coord_earth = np.zeros_like([3], dtype=float)
        # Altitude
        self.alt_pre = None  # Pressure altitude.
        self.alt_geop = None  # Geopotential altitude.
        # ANEMOMETRY
        self.TAS = None  # True Air Speed.
        self.CAS = None  # Calibrated Air Speed.
        self.EAS = None  # Equivalent Air Speed.
        self.IAS = None  # Indicated Air Speed.
        self.Mach = None  # Mach number.
        self.q_inf = None  # Dynamic pressure.
        self.alpha = None  # Angle of attack.
        self.beta = None  # Angle of sideslip.
        self.Dalpha_Dt = None  # d(alpha)/dt
        self.Dbeta_Dt = None  # d(beta)/dt
        # ATTITUDE (psi, theta, phi).
        self.euler_angles = np.zeros_like([3], dtype=float)
        self.quaternions = np.zeros_like([4], dtype=float)
        # VELOCITY
        # FIXME: Ground and absolute speed may differ depending on the
        # inertial reference frame considered.
        # Absolute body velocity.
        self.vel_body = np.zeros_like([3], dtype=float)
        # Absolute local horizon (NED) velocity
        self.vel_NED = np.zeros_like([3], dtype=float)
        self.gamma = None  # Flight path angle.
        # ANGULAR VELOCITY: (p, q, r)
        self.vel_ang = np.zeros_like([3], dtype=float)
        # TOTAL FORCES & MOMENTS (Body Axis)
        self.forces_body = np.zeros_like([3], dtype=float)
        self.moments_body = np.zeros_like([3], dtype=float)

        self.force_gravity = np.zeros_like([3], dtype=float)

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

    def set_initial_position(self, coordinates, mode='earth'):
        """
        Set the initial position of the aircraft.

        Parameters
        ----------
        coordinates: array_like, size=3
            Depending on the chosen mode the three coordinates are:
            * 'earth': (x_earth, y_earth, z_earth).
            * 'geographic': (lat, lon, h).
            * 'geocentric': (x_geo, y_geo, z_geo)
        mode: str, opt
            Kind of coordinates specified in `coordinates`.
            * 'earth': Earth axis are parallel to local horizon (NED) at the
             initial position but at h=0. Initial latitude and longitude
             will  be set to zero if this mode is chosen.
            * 'geographic': Latitude (geodetic), longitude and height above the
             reference ellipsoid (WGS84). Initial x_earth and y_earth will be
             set to zero if this mode is chosen.
            * 'geocentric': Geocentric reference frame rotates attached to
             Earth. Z is the Earth's rotation axis (pointing to north
             hemisphere and X is contained in the equatorial plane with
             longitude equal to zero. Initial x_earth and y_earth will be
             set to zero if this mode is chosen.
        """

        mode = mode.lower()
        # TODO: implement
        if mode == 'earth':
            pass
        elif mode == 'geographic':
            pass
        elif mode == 'geocentric':
            pass
        # self.coord_geographic = np.zeros([3], dtype=float)
        # self.coord_geocentric = np.zeros_like([3], dtype=float)
        # self.coord_earth = np.zeros_like([3], dtype=float)
        # self.alt_pre = None  # Pressure altitude.
        # self.alt_geop = None  # Geopotential altitude.

    # TODO: implement rest of inizialization (take trimmer into account).

class EulerFlatEarth(System):

    def __init__(self, integrator='dopri5', use_jac=False, **integrator_params):
        """
        Initialize the equations of the chosen model and selects the
        integrator. Check `scipy.integrate.ode` to see available integrators.

        If use_jac = True the jacobian of the equations is used for the
        integration.
        """
        super(EulerFlatEarth, self).__init__()
        # State vector must be initialized with set_initial_state_vector() method
        self.state_vector = None

        from pyfme.models.euler_flat_earth import lamceq, kaeq, kleq

        if use_jac:
            from pyfme.models.euler_flat_earth import lamceq_jac, kaeq_jac
            jac_LM_and_AM = lamceq_jac
            jac_att = kaeq_jac
            jac_nav = None  # not implemented
        else:
            jac_LM_and_AM = None
            jac_att = None
            jac_nav = None

        self._LM_and_AM_eqs = ode(lamceq, jac=jac_LM_and_AM)
        self._attitude_eqs = ode(kaeq, jac=jac_att)
        self._navigation_eqs = ode(kleq, jac=jac_nav)

        self._LM_and_AM_eqs.set_integrator(integrator, **integrator_params)
        self._attitude_eqs.set_integrator(integrator, **integrator_params)
        self._navigation_eqs.set_integrator(integrator, **integrator_params)

    def set_initial_state_vector(self, t0=0.0):
        """
        Set the initial values of the required variables
        """
        self.state_vector = np.array([
            self.u, self.v, self.w,
            self.p, self.q, self.r,
            self.theta, self.phi, self.psi,
            self.x_earth, self.y_earth, self.z_earth
        ])

        self._LM_and_AM_eqs.set_initial_value(y=self.state_vector[0:6], t=t0)
        self._attitude_eqs.set_initial_value(y=self.state_vector[6:9], t=t0)
        self._navigation_eqs.set_initial_value(y=self.state_vector[9:12], t=t0)

    def _propagate_state_vector(self, aircraft, dt):
        """
        Performs integration step for actual_time + dt and returns the state
        vector
        """
        mass = aircraft.mass
        inertia = aircraft.inertia
        forces = self.forces_body
        moments = self.moments_body

        t = self._LM_and_AM_eqs.t + dt

        self._LM_and_AM_eqs.set_f_params(mass, inertia, forces, moments)
        velocities = self._LM_and_AM_eqs.integrate(t)

        if self._LM_and_AM_eqs.successful():
            self._attitude_eqs.set_f_params(velocities[3:])
            attitude_angles = self._attitude_eqs.integrate(t)
        else:
            raise RuntimeError('Integration of Linear and angular momentum \
                                equations was not successful')

        if self._attitude_eqs.successful():
            self._navigation_eqs.set_f_params(velocities[0:3], attitude_angles)
            position = self._navigation_eqs.integrate(t)
        else:
            raise RuntimeError('Integration of attitude equations was not \
                                successful')

        if self._navigation_eqs.successful():
            self.state_vector[0:6] = velocities[:]
            self.state_vector[6:9] = attitude_angles[:]
            self.state_vector[9:12] = position[:]
        else:
            raise RuntimeError('Integration of navigation equations was not \
                                successful')

        return self.state_vector

    def propagate(self, aircraft, dt=0.01):
        """Propagate the state vector and update the rest of variables."""
        self._propagate_state_vector(aircraft, dt)
        # TODO: update the rest of variables.

