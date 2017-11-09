"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Simulation class
----------------
Select the simulation configuration based on a system (and its dynamic
model), environment and aircraft.

"""
import operator

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pyfme.utils.trimmer import trimming_cost_func


class Simulation(object):
    """
    Simulation class stores the simulation configuration, aircraft, system and
    environment. It provides methods for simulation running and results
    storing.

    Attributes
    ----------
    system : System
        System object with mathematical model of the dynamic system and
        integrator (ie. EulerFlatEarth)
    aircraft : Aircraft
        Aircraft model, where aerodynamics and forces are calculated
    environment : Environment
        Environment containing the atmosphere, gravity and wind models.
    controls : dict of callable
        Dictionary containing the control names as keys and functions of
        time as values.
    results : dict of lists
        Dictionary containing the variables that have been set to be saved
        during the simulation.
    """

    _default_save_vars = {
        'time': 'system.time',
        'T': 'environment.T',  # env
        'pressure': 'environment.p',
        'rho': 'environment.rho',
        'a': 'environment.a',
        'h': 'system.height',
        'Fx': 'aircraft.Fx',
        'Fy': 'aircraft.Fy',
        'Fz': 'aircraft.Fz',
        'Mx': 'aircraft.Mx',
        'My': 'aircraft.My',
        'Mz': 'aircraft.Mz',
        'TAS': 'aircraft.TAS',  # aircraft
        'Mach': 'aircraft.Mach',
        'q_inf': 'aircraft.q_inf',
        'alpha': 'aircraft.alpha',
        'beta': 'aircraft.beta',
        'rudder': 'aircraft.delta_rudder',
        'aileron': 'aircraft.delta_aileron',
        'elevator': 'aircraft.delta_elevator',
        'thrust': 'aircraft.delta_t',
        'x_earth': 'system.x_earth',  # system
        'y_earth': 'system.y_earth',
        'z_earth': 'system.z_earth',
        'height': 'system.height',
        'psi': 'system.psi',
        'theta': 'system.theta',
        'phi': 'system.phi',
        'u': 'system.u',
        'v': 'system.v',
        'w': 'system.w',
        'v_north': 'system.v_north',
        'v_east': 'system.v_east',
        'v_down': 'system.v_down',
        'p': 'system.p',
        'q': 'system.q',
        'r': 'system.r'
    }

    def __init__(self, aircraft, system, environment, save_vars=None):
        """
        Simulation object

        Parameters
        ----------
        aircraft : Aircraft
            Aircraft model
        system : System
            System model
        environment : Environment
            Environment model.
        save_vars : dict, opt
            Dictionary containing the names of the variables to be saved and
            the object and attribute where it is calculated. If not given, the
            ones set in `_defaul_save_vars` are used.
        """
        self.system = system

        # This wrap is necessary in order the respect the arguments passed
        # by the integration method: time, state (without self).
        update_fun = lambda time, state: self._time_step(time, state)
        self.system.model.set_forcing_terms(update_fun)

        save_fun = lambda time, state: self._save_time_step()
        self.system.model.set_solout(save_fun)

        self.aircraft = aircraft
        self.environment = environment

        self.controls = {}

        if not save_vars:
            self._save_vars = self._default_save_vars
        # Initialize results structure
        self.results = {name: [] for name in self._save_vars}

    def propagate(self, time):
        """Run the simulation by integrating the system until time t.

        Parameters
        ----------
        time : float
            Final time of the simulation

        Notes
        -----
        The propagation relies on the dense output of the integration
        method, so that the number and length of the time steps is
        automatically chosen.
        """

        self.environment.update(self.system)

        t0 = self.system.time

        controls0 = self._get_current_controls(t0)
        mass0, inertia0 = self.aircraft.mass, self.aircraft.inertia
        forces, moments = self.aircraft.calculate_forces_and_moments(
            self.system, self.environment, controls0)

        self.system.model.propagate(time, mass0, inertia0, forces, moments)

        # self.results is a dictionary of lists in order to append results
        # of each time step. Due to dense output of the integrator,
        # the number of time steps cannot be known in advance.
        # Once the integration has finished it can be transformed into a
        # DataFrame
        time = self.results.pop('time')
        self.results = pd.DataFrame(self.results, index=time)
        self.results.index.name = 'time'

    def _time_step(self, time, state):
        """Actions performed at each time step. This method is used as
        callback in the integration process.

        Parameters
        ----------
        time : float
            Current time value
        state : ndarray
            System state at the given time step

        Notes
        -----
        At each time step:
        * the full system state is updated given the model state,
        * the environment is updated given the current system,
        * the aircraft controls for the current time step are set
        * forces and moments for the current state, environment and controls
        are calculated.
        * the selected variables are saved.
        """
        self.system.model.time = time

        forces = self.aircraft.total_forces
        moments = self.aircraft.total_moments
        mass = self.aircraft.mass
        inertia = self.aircraft.inertia
        self.system.model.state = state
        self.system.set_full_system_state(mass, inertia, forces, moments)

        self.environment.update(self.system)

        # TODO: take into account that if the controls are not time
        # functions (ie. control system or AP are activated) the function
        # signature must be changed.
        controls = self._get_current_controls(time)

        forces, moments = self.aircraft.calculate_forces_and_moments(
            self.system,
            self.environment,
            controls
        )

        # self._save_time_step()

        return mass, inertia, forces, moments

    def _save_time_step(self):
        """Saves the selected variables for the current system, environment
        and aircraft.
        """
        for var_name, value_pointer in self._save_vars.items():
            self.results[var_name].append(
                operator.attrgetter(value_pointer)(self)
            )

    def _get_current_controls(self, time):
        """Get the control values for the current time step for the given
        input functions.

        Parameters
        ----------
        time : float
            Current time value.

        Returns
        -------
        controls : dict
            Control value for each control

        Notes
        -----
        Current controls are only a function of time in this kind of
        simulation (predefined inputs). However, if the AP is active,
        controls will be also function of the system state and environment.
        """
        c = {c_name: c_fun(time) for c_name, c_fun in self.controls.items()}
        return c
