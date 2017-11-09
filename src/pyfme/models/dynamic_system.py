# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic System
--------------

Dynamic system class to integrate initial value problems numerically serves
as base for implementation of dynamic systems.
"""
from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_ivp

from pyfme.models.acceleration import BodyAcceleration
from pyfme.models.angular_acceleration import BodyAngularAcceleration
from pyfme.models.angular_velocity import BodyAngularVelocity


class DynamicSystem:
    """Dynamic system class to integrate initial value problems numerically.
    Serves as base for implementation of dynamic systems.

    Attributes
    ----------
    state_vector : ndarray
        State vector.
    time : float
        Current integration time.

    Methods
    -------
    integrate(t_end, t_eval=None, dense_output=True)
        Integrate the system from current time to t_end.
    fun(t, x)
        Dynamic system equations
    """

    def __init__(self, t0, x0, method='Rk45', options=None):
        """ Dynamic system

        Parameters
        ----------
        t0 : float
            Initial time.
        x0 : array_like
            Initial state vector.
        method : str, opt
            Integration method. Accepts any method implemented in
            scipy.integrate.solve_ivp.
        options : dict
            Options for the selected method.
        """

        if options is None:
            options = {}
        self._state_vector = x0
        self._state_vector_dot = np.zeros_like(x0)
        self._time = t0

        self._method = method
        self._options = options

    @property
    def state_vector(self):
        return self._state_vector

    @property
    def state_vector_dot(self):
        return self._state_vector_dot

    @property
    def time(self):
        return self._time

    def integrate(self, t_end, t_eval=None, dense_output=True):
        """Integrate the system from current time to t_end.

        Parameters
        ----------
        t_end : float
            Final time.
        t_eval : array_like, opt
            Times at which to store the computed solution, must be sorted
            and lie within current time and t_end. If None (default), use
            points selected by a solver.
        dense_output: bool, opt
            Whether to compute a continuous solution. Default is True.

        Returns
        -------
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Solution values at t.
        sol : Bunch object with the following fields defined:
            t : ndarray, shape (n_points,)
                Time points.
            y : ndarray, shape (n, n_points)
                Solution values at t.
            sol : OdeSolution or None
                Found solution as OdeSolution instance, None if dense_output
                was set to False.
            t_events : list of ndarray or None
                Contains arrays with times at each a corresponding event was
                detected, the length of the list equals to the number of
                events. None if events was None.
            nfev : int
                Number of the system rhs evaluations.
            njev : int
                Number of the Jacobian evaluations.
            nlu : int
                Number of LU decompositions.
            status : int
                Reason for algorithm termination:
                -1: Integration step failed.
                0: The solver successfully reached the interval end.
                1: A termination event occurred.
            message : string
                Verbal description of the termination reason.
            success : bool
            True if the solver reached the interval end or a termination event
             occurred (status >= 0).
        """
        # TODO: intended to return the whole integration history
        # How dos it update the full system?
        x0 = self.state_vector
        t_ini = self.time

        t_span = (t_ini, t_end)
        method = self._method

        # TODO: prepare to use jacobian in case it is defined
        sol = solve_ivp(self.fun, t_span, x0, method=method, t_eval=t_eval,
                        dense_output=dense_output, **self._options)

        self._time = sol.t[-1]
        self._state_vector = sol.y[:, -1]

        return sol.t, sol.y, sol

    def time_step(self, dt):
        """Integrate the system from current time to t_end.

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        y : ndarray, shape (n)
            Solution values at t_end.
        """

        x0 = self.state_vector
        t_ini = self.time

        t_span = (t_ini, t_ini + dt)
        method = self._method

        # TODO: prepare to use jacobian in case it is defined
        sol = solve_ivp(self.fun_wrapped, t_span, x0, method=method,
                        **self._options)

        if sol.status == -1:
            raise RuntimeError(f"Integration did not converge at t={t_ini}")

        self._time = sol.t[-1]
        self._state_vector = sol.y[:, -1]

        return self._state_vector

    @abstractmethod
    def fun(self, t, x):
        """ Right-hand side of the system (dy / dt = f(t, y)). The calling
        signature is fun(t, x). Here t is a scalar and there are two options
        for ndarray y.
        It can either have shape (n,), then fun must return array_like with
        shape (n,). Or alternatively it can have shape (n, k), then fun must
        return array_like with shape (n, k), i.e. each column corresponds to a
        single column in y. The choice between the two options is determined
        by vectorized argument (see below). The vectorized implementation
        allows faster approximation of the Jacobian by finite differences
        (required for stiff solvers).
        """
        raise NotImplementedError

    def fun_wrapped(self, t, x):
        # First way that comes to my mind in order to store the derivates
        # that are useful for full_state calculation
        state_dot = self.fun(t, x)
        self._state_vector_dot = state_dot
        return state_dot


class AircraftDynamicSystem(DynamicSystem):

    def __init__(self, t0, full_state, update, method='Rk45', options=None):
        x0 = self._get_state_vector_from_full_state(full_state)
        self.full_state = self._adapt_full_state_to_dynamic_system(full_state)

        super().__init__(t0, x0, method=method, options=options)

        self.update_simulation = update

    @abstractmethod
    def _adapt_full_state_to_dynamic_system(self, full_state):
        raise NotImplementedError

    @abstractmethod
    def _update_full_system_state_from_state(self, state, state_dot):
        raise NotImplementedError

    @abstractmethod
    def _get_state_vector_from_full_state(self, full_state):
        raise NotImplementedError

    @abstractmethod
    def trim_fun(self, full_state, environment, aircraft, controls):
        raise NotImplementedError

    def time_step(self, dt):
        super().time_step(dt)
        # Now self.state_vector and state_vector_dot are updated
        self._update_full_system_state_from_state(self.state_vector,
                                                  self.state_vector_dot)

        return self.full_state


class AircraftState:

    def __init__(self, position, attitude, velocity, angular_vel=None,
                 acceleration=None, angular_accel=None):

        self.position = position
        self.attitude = attitude
        self.velocity = velocity

        if angular_vel is None:
            angular_vel = BodyAngularVelocity(0, 0, 0, attitude)
        if acceleration is None:
            acceleration = BodyAcceleration(0, 0, 0, attitude)
        if angular_accel is None:
            angular_accel = BodyAngularAcceleration(0, 0, 0, attitude)

        self.angular_vel = angular_vel
        self.acceleration = acceleration
        self.angular_accel = angular_accel
