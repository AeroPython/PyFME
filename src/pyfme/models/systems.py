"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic Systems
---------------

"""

import numpy as np
from scipy.integrate import ode

from pyfme.aircrafts.aircraft import Aircraft
from pyfme.environment.environment import Environment
from pyfme.utils.altimetry import geometric2geopotential
from pyfme.utils.anemometry import tas2cas, tas2eas


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
        self.turn_rate = None  # d(psi)/dt
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

    def set_initial_flight_conditions(self, lat, lon, h, TAS,
                                      environment: Environment,
                                      gamma=0, turn_rate=0):

        self.coord_geographic[0] = lat
        self.coord_geographic[1] = lon
        self.coord_geographic[2] = h
        # TODO: Conversion to geocentric coordinates
        # self.coord_geocentric =

        self.alt_pre = h
        self.alt_geop = geometric2geopotential(h)

        environment.update(self)
        rho = environment.rho
        a = environment.a
        p = environment.p
        self.Mach = TAS / a
        self.q_inf = 0.5 * rho * TAS**2
        self.TAS = TAS
        self.CAS = tas2cas(TAS, p, rho)
        self.EAS = tas2eas(TAS, rho)
        # self.IAS =

        self.Dalpha_Dt = 0  # d(alpha)/dt
        self.Dbeta_Dt = 0  # d(beta)/dt

        self.gamma = gamma
        self.turn_rate = turn_rate

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

        self._ode_lamceq.set_initial_value(y=self.state_vector[0:6], t=t0)
        self._ode_kaqeq.set_initial_value(y=self.state_vector[6:9], t=t0)
        self._ode_kleq.set_initial_value(y=self.state_vector[9:12], t=t0)

    def _propagate_state_vector(self, aircraft, dt):
        """
        Performs integration step for actual_time + dt and returns the state
        vector
        """
        mass = aircraft.mass
        inertia = aircraft.inertia
        forces = self.forces_body
        moments = self.moments_body

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
        """Propagate the state vector and update the rest of variables."""
        self._propagate_state_vector(aircraft, dt)
        # TODO: update the rest of variables.

