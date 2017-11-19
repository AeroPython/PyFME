"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Simulation class
----------------
Select the simulation configuration based on a system (and its dynamic
model), environment and aircraft.

"""
import copy
import operator

import pandas as pd
import tqdm


class Simulation:
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
        # environment
        'temperature': 'environment.T',
        'pressure': 'environment.p',
        'rho': 'environment.rho',
        'a': 'environment.a',
        # aircraft
        'Fx': 'aircraft.Fx',
        'Fy': 'aircraft.Fy',
        'Fz': 'aircraft.Fz',

        'Mx': 'aircraft.Mx',
        'My': 'aircraft.My',
        'Mz': 'aircraft.Mz',

        'TAS': 'aircraft.TAS',
        'Mach': 'aircraft.Mach',
        'q_inf': 'aircraft.q_inf',

        'alpha': 'aircraft.alpha',
        'beta': 'aircraft.beta',

        'rudder': 'aircraft.delta_rudder',
        'aileron': 'aircraft.delta_aileron',
        'elevator': 'aircraft.delta_elevator',
        'thrust': 'aircraft.delta_t',
        # system
        'x_earth': 'system.full_state.position.x_earth',
        'y_earth': 'system.full_state.position.y_earth',
        'z_earth': 'system.full_state.position.z_earth',

        'height': 'system.full_state.position.height',

        'psi': 'system.full_state.attitude.psi',
        'theta': 'system.full_state.attitude.theta',
        'phi': 'system.full_state.attitude.phi',

        'u': 'system.full_state.velocity.u',
        'v': 'system.full_state.velocity.v',
        'w': 'system.full_state.velocity.w',

        'v_north': 'system.full_state.velocity.v_north',
        'v_east': 'system.full_state.velocity.v_east',
        'v_down': 'system.full_state.velocity.v_down',

        'p': 'system.full_state.angular_vel.p',
        'q': 'system.full_state.angular_vel.q',
        'r': 'system.full_state.angular_vel.r'
    }

    def __init__(self, aircraft, system, environment, controls, dt=0.01,
                 save_vars=None):
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
            ones set in `_default_save_vars` are used.
        """
        self.system = copy.deepcopy(system)
        self.aircraft = copy.deepcopy(aircraft)
        self.environment = copy.deepcopy(environment)

        self.system.update_simulation = self.update

        self.controls = controls

        self.dt = dt

        if not save_vars:
            self._save_vars = self._default_save_vars
        # Initialize results structure
        self.results = {name: [] for name in self._save_vars}

    @property
    def time(self):
        return self.system.time

    def update(self, time, state):
        self.environment.update(state)

        controls = self._get_current_controls(time)

        self.aircraft.calculate_forces_and_moments(
            state,
            self.environment,
            controls
        )
        return self

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
        dt = self.dt
        half_dt = self.dt/2

        bar = tqdm.tqdm(total=time, desc='time', initial=self.system.time)

        # To deal with floating point issues we cannot check equality to
        # final time to finish propagation
        time_plus_half_dt = time + half_dt
        while self.system.time + dt < time_plus_half_dt:
            t = self.system.time
            self.environment.update(self.system.full_state)
            controls = self._get_current_controls(t)
            self.aircraft.calculate_forces_and_moments(self.system.full_state,
                                                       self.environment,
                                                       controls)
            self.system.time_step(dt)
            self._save_time_step()
            bar.update(dt)

        bar.close()

        results = pd.DataFrame(self.results)
        results.set_index('time', inplace=True)

        return results

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
