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
        'x_earth': 'system.x_earth',  # system
        'y_earth': 'system.y_earth',
        'z_earth': 'system.z_earth',
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
        callback = lambda time, state: self._time_step(time, state)
        self.system.model.set_solout(callback)

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
        # dict of ndarrays
        self.results = {n: np.asarray(v) for n, v in self.results.items()}

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

        forces, moment = self.aircraft.calculate_forces_and_moments(
            self.system,
            self.environment,
            controls
        )

        self.system.model.set_forcing_terms(mass, inertia, forces, moments)

        self._save_time_step()

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

    def trim_aircraft(self, geodetic_initial_pos, TAS, gamma, turn_rate,
                      initial_controls, psi=0, exclude_controls=[], verbose=0):
        """Finds a combination of values of the state and control variables
        that correspond to a steady-state flight condition.

        Steady-state aircraft flight is defined as a condition in which all
        of the motion variables are constant or zero. That is, the linear and
        angular velocity components are constant (or zero), thus all
         acceleration components are zero.

        Parameters
        ----------
        geodetic_initial_pos : arraylike, shape(3)
            (Latitude, longitude, height)
        TAS : float
            True Air Speed (m/s).
        gamma : float, optional
            Flight path angle (rad).
        turn_rate : float, optional
            Turn rate, d(psi)/dt (rad/s).
        initial_controls : dict
            Initial value guess for each control.
        psi : float, opt
            Initial yaw angle (rad).
        exclude_controls : list, optional
            List with controls not to be trimmed. If not given, every control
            is considered in the trim process.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:
                * 0 (default) : work silently.
                * 1 : display a termination report.
                * 2 : display progress during iterations (not supported by 'lm'
                  method).

        Notes
        -----
        See section 3.4 in [1] for the algorithm description.
        See section 2.5 in [1] for the definition of steady-state flight
        condition.

        References
        ----------
        .. [1] Stevens, BL and Lewis, FL, "Aircraft Control and Simulation",
            Wiley-lnterscience.
        """

        system = self.system
        env = self.environment
        ac = self.aircraft

        # Initialize state
        system.set_initial_state(
            geodetic_coordinates=geodetic_initial_pos,
            euler_angles=np.array([0, 0, psi])
                                 )
        # Update environment for the current state
        env.update(system)

        # Initialize alpha and beta
        # TODO: improve initialization method
        alpha0 = 0.05
        beta0 = 0.001 * np.sign(turn_rate)

        # For the current alpha, beta, TAS and env, set the aerodynamics of
        # the aircraft (q_inf, CAS, EAS...)
        ac._calculate_aerodynamics_2(TAS, alpha0, beta0, env)

        # Initialize controls
        for control in ac.controls:
            if control not in initial_controls:
                raise ValueError(
                    "Control {} not given in initial_controls: {}".format(
                        control, initial_controls)
                )
            else:
                ac.controls[control] = initial_controls[control]

        # Select the controls that will be trimmed
        controls_to_trim = list(ac.controls.keys() - exclude_controls)

        # Set the variables for the optimization
        initial_guess = [alpha0, beta0]
        for control in controls_to_trim:
            initial_guess.append(initial_controls[control])

        # Set bounds for each variable to be optimized
        lower_bounds = [-0.5, -0.25]  # Alpha and beta upper bounds.
        upper_bounds = [+0.5, +0.25]  # Alpha and beta lower bounds.
        for ii in controls_to_trim:
            lower_bounds.append(ac.control_limits[ii][0])
            upper_bounds.append(ac.control_limits[ii][1])
        bounds = (lower_bounds, upper_bounds)

        args = (system, ac, env, controls_to_trim, gamma, turn_rate)

        # Trim
        results = least_squares(trimming_cost_func,
                                x0=initial_guess,
                                args=args,
                                verbose=verbose,
                                bounds=bounds)
