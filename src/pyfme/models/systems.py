"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic Systems
---------------

Dynamic system implements an abstract class to be inherited by all the
models.

System is a more generic object that wraps every system and keeps updated
variables that may not be present in the dynamic system but may be used by
other members of the simulation, such as the environment or the aircraft.
"""
from abc import abstractmethod, abstractstaticmethod

import numpy as np
from scipy.integrate import ode

from pyfme.utils.coordinates import body2hor, hor2body


class System(object):
    """Generic system object containing and wrapping a model that keeps updated
    variables that may not be present in the model but may be inferred from it.
    
    Attributes
    ----------
    model : DynamicSystem
        System's model containing the equations and state vector.
    full_system_state_names : tuple of str
        Names of all variables that are tracked in the system. All of them 
        are listed as class attributes below.
    geodetic_coordinates : ndarray, shape(3)
        (lat [rad], lon [rad], height [m])
    geocentric_coordinates : ndarray, shape(3)
        (x_geo [m], y_geo [m], z_geo [m])
    earth_coordinates : ndarray, shape(3)
        (x_earth [m], y_earth [m], z_earth [m])
    euler_angles : ndarray, shape(3)
        (theta [rad], phi [rad], psi [rad])
    quaternions : ndarray, shape(4)
        (q0, q1, q2, q3)
    vel_body : ndarray, shape(3)
        (u [m/s], v [m/s], w [m/s])
    vel_NED : ndarray, shape(3)
        (v_north [m/s], v_east [m/s], v_down [m/s])
    vel_ang : ndarray, shape(3)
        (p [rad/s], q [rad/s], r [rad/s])
    accel_body : ndarray, shape(3)
        (u_dot [m/s²], v_dot [m/s²], w_dot [m/s²])
    accel_NED : ndarray, shape(3)
        (VN_dot [m/s²], VE_dot [m/s²], VD_dot [m/s²])
    accel_ang : ndarray, shape(3)
        (p_dot [rad/s²], q_dot [rad/s²], r_dot [rad/s²])

    """

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
        """System

        Parameters
        ----------
        model : DynamicSystem
        System's model containing the equations and state vector.
        """

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
        return self.euler_angles[2]

    @property
    def theta(self):
        return self.euler_angles[0]

    @property
    def phi(self):
        return self.euler_angles[1]

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
    def u_dot(self):
        return self.accel_body[0]

    @property
    def v_dot(self):
        return self.accel_body[1]

    @property
    def w_dot(self):
        return self.accel_body[2]

    @property
    def v_north_dot(self):
        return self.accel_NED[0]

    @property
    def v_east_dot(self):
        return self.accel_NED[1]

    @property
    def v_down_dot(self):
        return self.accel_NED[2]

    @property
    def p_dot(self):
        return self.accel_ang[0]

    @property
    def q_dot(self):
        return self.accel_ang[1]

    @property
    def r_dot(self):
        return self.accel_ang[2]

    @property
    def time(self):
        return self.model.time

    @time.setter
    def time(self, value):
        self.model.time = value

    def set_initial_state(self,
                          geodetic_coord=None, geocentric_coord=None,
                          euler_angles=None,
                          quaternions=None,
                          vel_body=None, vel_NED=None,
                          vel_ang=None,
                          accel_body=None, accel_NED=None,
                          accel_ang=None):
        """Initialize the system state and the model state.

        Parameters
        ----------
        geodetic_coord : ndarray, shape(3), opt
            (lat [rad], lon [rad], height [m]). Only if geodetic_coord are
            not given.
        geocentric_coord : ndarray, shape(3), opt
            (x_geo [m], y_geo [m], z_geo [m]). Only if geocentric_coord are
            not given.
        euler_angles : ndarray, shape(3), opt
            (psi [rad], theta [rad], phi [rad]). Only if quaternions are not
            given.
        quaternions : ndarray, shape(4), opt
            (q0, q1, q2, q3). Only if euler_angles are not given.
        vel_body : ndarray, shape(3), opt
            (u [m/s], v [m/s], w [m/s]). Only if vel_NED is not given.
        vel_NED : ndarray, shape(3), opt
            (v_north [m/s], v_east [m/s], v_down [m/s]). Only if vel_body is
            not given.
        vel_ang : ndarray, shape(3), opt
            (p [rad/s], q [rad/s], r [rad/s])
        accel_body : ndarray, shape(3), opt
            (u_dot [m/s²], v_dot [m/s²], w_dot [m/s²]). Only if accel_NED is
            not given.
        accel_NED : ndarray, shape(3), opt
            (VN_dot [m/s²], VE_dot [m/s²], VD_dot [m/s²]). Only if
            accel_body is not given.
        accel_ang : ndarray, shape(3), opt
            (p_dot [rad/s²], q_dot [rad/s²], r_dot [rad/s²])
        """

        # Set position
        if (geodetic_coord is not None) and (geocentric_coord is not None):
            raise ValueError("Provide only geodetic or geocentric, not both "
                             "at the same time")

        elif geodetic_coord is not None:
            self.geodetic_coordinates = geodetic_coord
            # TODO: geodetic to geocentric (cartopy?)
            # self.geocentric_coordinates = transformation
        elif geocentric_coord is not None:
            self.geocentric_coordinates = geocentric_coord
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
            raise ValueError("Provide only euler angles or quaternions, "
                             "not both at the same time")
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
            raise ValueError("Provide only vel_body or vel_NED, not both at "
                             "the same time")
        elif vel_body is not None:
            self.vel_body = vel_body
            self.vel_NED = body2hor(vel_body, self.theta, self.phi, self.psi)
        elif vel_NED is not None:
            self.vel_NED = vel_NED
            self.vel_body = hor2body(vel_NED, self.theta, self.phi, self.psi)
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
            raise ValueError("Provide only accel_body or accel_NED, not both "
                             "at the same time")
        elif accel_body is not None:
            self.accel_body = accel_body
            self.accel_NED = body2hor(accel_body, self.theta, self.phi,
                                      self.psi)
        elif accel_NED is not None:
            self.accel_NED = accel_NED
            self.accel_body = hor2body(accel_NED, self.theta, self.phi,
                                       self.psi)
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

    # TODO: it's more explicit to pass the state vector here instead of
    # depending on its update in the model before.
    def set_full_system_state(self, mass, inertia, forces, moments):
        """ Updates the full system update based on the current state vector

        Parameters
        ----------
        mass : float
            Aircraft mass
        inertia : ndarray, shape(3, 3)
            Inertia tensor of the aircraft
        forces : ndarray, shape(3)
            Aircraft forces
        moments : ndarray, shape(3)
            Aircraft moments
        """
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
        f = (lambda time, state_vector, update_fun:
            self.dynamic_system_equations(time, state_vector, update_fun))

        self._ode = ode(f, self._jacobian)

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

    def _set_fun_forcing_terms(self, update_f):
        self._ode.set_f_params(update_f)

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
        # if not self._ode._integrator.solout:
        #     raise ValueError("A callback to the model must be given in order "
        #                      "to update the system, environment and aircraft "
        #                      "at each time step. Also to save the results."
        #                      )

        # Sets the final time of the integration
        t = self._ode.t + dt

        # This only affects the first time step: this update will be done by
        # the callback function at every integration step after updating
        # mass, inertia, forces and moments
        # self.set_forcing_terms(mass, inertia, forces, moments)

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
