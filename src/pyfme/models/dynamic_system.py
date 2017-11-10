# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Dynamic System & Aircraft Dynamic System
----------------------------------------

Dynamic system class to integrate initial value problems numerically serves
as base for implementation of dynamic systems.

The Aircraft Dynamic Systems extends the Dynamic System taking into account
the Aircraft State.
"""
from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


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

    def __init__(self, t0, x0, method='RK45', options=None):
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
        # TODO: this mehtod is intended to return the whole integration history
        # meanwhile, only time_step is called
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
        # First way that comes to my mind in order to store the derivatives
        # that are useful for full_state calculation
        state_dot = self.fun(t, x)
        self._state_vector_dot = state_dot
        return state_dot


class AircraftDynamicSystem(DynamicSystem):
    """Aircraft's Dynamic System

    Attributes
    ----------
    full_state : AircraftState
        Current aircraft state.
    update_simulation : fun
        Function that updates environment and aircraft in order to get
        proper forces, moments and mass properties for integration steps.

    Methods
    -------
    steady_state_trim_fun
    time_step
    """
    def __init__(self, t0, full_state, method='RK45', options=None):
        """Aircraft Dynamic system initialization.

        Parameters
        ----------
        t0 : float
            Initial time (s).
        full_state : AircraftState
            Initial aircraft's state.
        method : str, opt
            Integration method. Any method included in
            scipy.integrate.solve_ivp can be used.
        options : dict, opt
            Options accepted by the integration method.
        """
        x0 = self._get_state_vector_from_full_state(full_state)
        self.full_state = self._adapt_full_state_to_dynamic_system(full_state)

        super().__init__(t0, x0, method=method, options=options)

        self.update_simulation = None

    @abstractmethod
    def _adapt_full_state_to_dynamic_system(self, full_state):
        """Transforms the given state to the one used in the
        AircraftDynamicSystem in order to initialize dynamic's system
        initial state.
        For example, the full state given may be using quaternions for
        attitude representation, but the Aircraft dynamic system may
        propagate Euler angles.

        Parameters
        ----------
        full_state : AircraftState

        """
        raise NotImplementedError

    @abstractmethod
    def _update_full_system_state_from_state(self, state, state_dot):
        """Updates full system's state (AircraftState) based on the
        implemented dynamic's system state vector and derivative of system's
        state vector (output of system's equations dx/dt=f(x, t))

        Parameters
        ----------
        state : array_like
            State vector.
        state_dot : array_like
            Derivative of state vector.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_state_vector_from_full_state(self, full_state):
        """Gets the state vector given the full state.

        Parameters
        ----------
        full_state : AircraftState
            Aircraft's full state

        Returns
        -------
        state_vector : ndarray
            State vector.
        """
        raise NotImplementedError

    @abstractmethod
    def steady_state_trim_fun(self, full_state, environment, aircraft,
                              controls):
        """Output from linear and angular momentum conservation equations:
        ax, ay, az, p, q, r, which must be zero after the steady state
        trimming process

        Parameters
        ----------
        full_state : AircraftState
            Full aircraft state.
        environment : Environment
            Environment in which the aircraft is being trimmed.
        aircraft : Aircraft
            Aircraft being trimmed.
        controls : dict
            Controls of the aircraft being trimmed.

        Returns
        -------
        rv : ndarray
            Output from linear and angular momentum conservation equations:
            ax, ay, az, p, q, r.
        """
        raise NotImplementedError

    def time_step(self, dt):
        """Perform an integration time step

        Parameters
        ----------
        dt : float
            Time step for integration

        Returns
        -------
        full_state : AircraftState
            Aircraft's state after integration time step
        """
        super().time_step(dt)
        # Now self.state_vector and state_vector_dot are updated
        self._update_full_system_state_from_state(self.state_vector,
                                                  self.state_vector_dot)

        return self.full_state
