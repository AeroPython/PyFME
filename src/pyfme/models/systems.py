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

    full_system_state_names = ('geodetic_coordinates',
                               'geocentric_coordinates',
                               'earth_coordinates',
                               'euler_angles',
                               'quaternions',
                               'vel_body',
                               'vel_NED',
                               'vel_ang',
                               'accel_body',
                               'accel_NED',
                               'accel_ang')

    def __init__(self, model):

        # Dynamic system
        self.model = model

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
        return self.model.time

    # TODO: break in several methods: position, attitude...
    def set_initial_state(self, geodetic_coordinates=None,
            geocentric_coordinates=None, euler_angles=None, quaternions=None,
            vel_body=None, vel_NED=None,  vel_ang=None, accel_body=None,
            accel_NED=None, accel_ang=None):

        # Set position
        if (geodetic_coordinates is not None) and (geocentric_coordinates is
                                                   not None):
            # TODO: complete error (two values given)
            raise ValueError()

        elif geodetic_coordinates is not None:
            self.geodetic_coordinates = geodetic_coordinates
            # TODO: geodetic to geocentric (cartopy?)
            # self.geocentric_coordinates = transformation
        elif geocentric_coordinates is not None:
            self.geocentric_coordinates = geocentric_coordinates
            # TODO: geocentric to geodetic (cartopy?)
        else:
            # no values for position
            self.geodetic_coordinates = np.zeros(3)
            # TODO: geodetic to geocentric (cartopy?)
            # self.geocentric_coordinates = transformation
            pass

        self.earth_coordinates[2] = self.geodetic_coordinates[2]

        # Set attitude
        if (quaternions is not None) and (euler_angles is not None):
            # TODO: complete error (two values given)
            raise ValueError()
        elif quaternions is not None:
            self.quaternions = quaternions
            # TODO: quaternions to euler
            # self.euler = transformation
        elif euler_angles is not None:
            self.euler_angles = euler_angles
            # TODO: euler angles to quaternions
            # self.quaternions = transformation
        else:
            self.euler_angles = np.zeros(3)
            # TODO: euler angles to quaternions
            # self.quaternions = transformation

        # Set velocity
        if (vel_body is not None) and (vel_NED is not None):
            # TODO: complete error (two values given)
            raise ValueError()
        elif vel_body is not None:
            self.vel_body = vel_body
            # TODO: body to horizon
            # self.vel_NED = transformation
        elif vel_NED is not None:
            self.vel_NED = vel_NED
            # TODO: horizon to body
            # self.vel_body = transformation
        else:
            self.vel_body = np.zeros(3)
            self.vel_NED = np.zeros(3)

        # Set angular velocity
        if vel_ang is not None:
            self.vel_ang = vel_ang
        else:
            self.vel_ang = np.zeros(3)

        # Set accelerations
        if (accel_body is not None) and (accel_NED is not None):
            # TODO: complete error (two values given)
            raise ValueError()
        elif accel_body is not None:
            self.accel_body = accel_body
            # TODO: body to horizon
            # self.acc_NED = transformation
        elif accel_NED is not None:
            self.accel_NED = accel_NED
            # TODO: horizon to body
            # self.acc_body = transformation
        else:
            self.accel_body = np.zeros(3)
            self.accel_NED = np.zeros(3)

        # Set angular velocity
        if vel_ang is not None:
            self.vel_ang = vel_ang
        else:
            self.vel_ang = np.zeros(3)

        # Set angular accelerations
        if accel_ang is not None:
            self.accel_ang = accel_ang
        else:
            self.accel_ang = np.zeros(3)

        state = self.model.full_system_state_to_dynamic_system_state(self)
        self.model.set_initial_state(state)

    def set_full_system_state(self, mass, inertia, forces, moments):
        rv = self.model.dynamic_system_state_to_full_system_state(
            mass, inertia, forces, moments)

        for name in self.full_system_state_names:
            self.__setattr__(name, rv[name])


class DynamicSystem(object):
    """Dynamic system abstract class.

    Attributes
    ----------
    state : ndarray
        State vector.
    time : float
        Current time of the simulation.
    _ode : scipy.integrate.ode
        Ordinary Differential Equation based on function definded in
        `dynamic_system_equations` method and jacobian in
        `dynamic_system_jacobian` method.
    """

    def __init__(self, n_states, use_jacobian=False, integrator=None,
                 **integrator_params):
        """Dynamic system initialization

        Parameters
        ----------
        n_states : int
            Number of states of the dynamical system.
        use_jacobian: bool, opt
            Whether to use jacobian of the system's model during the
            integration or not.
        integrator : str, opt
            Integrator to use by scipy.integrate.ode. By default, dopri5 is
            used. Check scipy doc in order to list all possibilities.
        **integrator_params : dict, opt
            Other integrator params passed as kwargs.
        """

        self.time = 0.
        # Allocate state vector
        self.state = np.empty(n_states)

        # Set the jacobian if it is implemented in the model
        if use_jacobian:
            self._jacobian = self.dynamic_system_jacobian
            self.set_forcing_terms = self._set_fun_and_jac_forcing_terms
        else:
            self._jacobian = None
            self.set_forcing_terms = self._set_fun_forcing_terms

        # ODE setup
        self._ode = ode(self.dynamic_system_equations, self._jacobian)

        if integrator is None:
            integrator = 'dopri5'

        # TODO: carefully review integrator parameters such as nsteps
        self._ode.set_integrator(integrator, nsteps=10000, **integrator_params)

    def set_initial_state(self, state):
        """Sets the initial state for the integration

        Parameters
        ----------
        state : ndarray
            State vector
        """
        self.state = state
        self._ode.set_initial_value(self.state)

    def _set_fun_forcing_terms(self, mass, inertia, forces, moments):
        self._ode.set_f_params(mass, inertia, forces, moments)

    def _set_jac_forcing_terms(self, mass, inertia, forces, moments):
        self._ode.set_jac_params(mass, inertia, forces, moments)

    def _set_fun_and_jac_forcing_terms(self, mass, inertia, forces, moments):
        self._set_fun_and_jac_forcing_terms(mass, inertia, forces, moments)
        self._set_jac_forcing_terms(mass, inertia, forces, moments)

    def set_solout(self, fun):
        """Set callback for scipy.integrate.ode solver

        Parameters
        ----------
        fun : callable
            Function to be called at each time step during the integration.
            It must be in charge of updating the whole system, environment,
            controls, forces, moments, mass, inertia...
        """
        self._ode.set_solout(fun)

    def propagate(self, dt, mass, inertia, forces, moments):
        """ Perform the integration from the current time step during time dt.

        Parameters
        ----------
        dt : float
            Time for the integration.
        mass : float
            Current aircraft mass (initial time step).
        inertia : ndarray, shape (3, 3)
            Current aircraft inertia (initial time step).
        forces : ndarray, shape(3)
            Current aircraft forces (initial time step).
        moments : ndarray, shape(3)
            Current aircraft moments (initial time step).

        Returns
        -------
        state : ndarray
            Final state if integration is successful.

        Raises
        ------
        RunTimeError if integration is not successful.
        """

        # Checks that a callback for updating environment and aircraft has
        # been defined previous to integration
        if not self._ode._integrator.solout:
            raise ValueError("A callback to the model must be given in order "
                             "to update the system, environment and aircraft "
                             "at each time step. Also to save the results."
                             )

        # Sets the final time of the integration
        t = self._ode.t + dt

        # This only affects the first time step: this update will be done by
        # the callback function at every integration step after updating
        # mass, inertia, forces and moments
        self.set_forcing_terms(mass, inertia, forces, moments)

        # Perform the integration
        self.state = self._ode.integrate(t)

        if self._ode.successful():
            return self.state
        else:
            raise RuntimeError("Error during integration")

    @abstractmethod
    def dynamic_system_state_to_full_system_state(self, mass, inertia,
                                                  forces, moments):
        raise NotImplementedError

    @abstractmethod
    def full_system_state_to_dynamic_system_state(self, full_system):
        raise NotImplementedError

    @abstractmethod
    def trim_system_to_dynamic_system_state(self, full_system):
        raise NotImplementedError

    @abstractstaticmethod
    def dynamic_system_equations(time, state_vector, mass, inertia, forces,
                                 moments):
        raise NotImplementedError

    @abstractstaticmethod
    def dynamic_system_jacobian(state_vector, mass, inertia, forces, moments):
        raise NotImplementedError
