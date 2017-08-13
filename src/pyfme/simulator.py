"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Simulation class
----------------

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
    """

    def __init__(self, aircraft, system, environment):
        """
        Simulation objects.

        Parameters
        ----------
        aircraft : Aircraft
            Aircraft.
        system : System
            System.
        environment : Environment
            Environment.
        """
        self.system = system

        callback = lambda time, state: self.time_step(time, state)
        self.system.model._ode.set_solout(callback)

        self.aircraft = aircraft
        self.environment = environment

        self.controls = {}

        self.vars_to_save = {
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

        self.results = {name: [] for name in self.vars_to_save}

    def propagate(self, time):

        self.environment.update(self.system)

        t0 = self.system.time

        controls0 = self.get_current_controls(t0)
        mass0, inertia0 = self.aircraft.mass, self.aircraft.inertia
        forces, moments = self.aircraft.calculate_forces_and_moments(
            self.system, self.environment, controls0)

        self.system.model.propagate(time, mass0, inertia0, forces, moments)

    def time_step(self, time, state):
        forces = self.aircraft.total_forces
        moments = self.aircraft.total_moments
        mass = self.aircraft.mass
        inertia = self.aircraft.inertia

        self.system.model.state = state
        self.system.set_full_system_state(mass, inertia, forces, moments)

        self.environment.update(self.system)

        controls = self.get_current_controls(time)

        forces, moment = self.aircraft.calculate_forces_and_moments(
            self.system,
            self.environment,
            controls
        )

        self.system.model.set_forcing_terms(mass, inertia, forces, moments)

        print(time, state)

        self.save_time_step()

    def save_time_step(self):

        for var_name, value_pointer in self.vars_to_save.items():
            self.results[var_name].append(operator.attrgetter(value_pointer)(
                self))

    def get_current_controls(self, time):
        c = {c_name: c_fun(time) for c_name, c_fun in self.controls.items()}
        return c

    def trim_aircraft(self, geodetic_initial_pos, TAS, gamma, turn_rate,
                      initial_controls, exclude_controls=[], verbose=0):
        """Finds a combination of values of the state and control variables that
        correspond to a steady-state flight condition. Steady-state aircraft flight
        can be defined as a condition in which all of the motion variables are
        constant or zero. That is, the linear and angular velocity components are
        constant (or zero), thus all acceleration components are zero.

        Parameters
        ----------
        geodetic_initial_pos : ndarray, shape(3)
            (Latitude, longitude, height)
        TAS : float
            True Air Speed (m/s).
        gamma : float, optional
            Flight path angle (rad).
        turn_rate : float, optional
            Turn rate, d(psi)/dt (rad/s).
        initial_controls : dict
            Initial value guess for each control.
        exclude_controls : list, optional
            List with controls not to be trimmed. If not given,
            every control is considered fixed.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:

                * 0 (default) : work silently.
                * 1 : display a termination report.
                * 2 : display progress during iterations (not supported by 'lm'
                  method).

        Notes
        -----
        See section 3.4 in [1] for the algorithm description.
        See section 2.5 in [1] for the definition of steady-state flight condition.

        References
        ----------
        .. [1] Stevens, BL and Lewis, FL, "Aircraft Control and Simulation",
            Wiley-lnterscience.
        """

        system = self.system
        env = self.environment
        ac = self.aircraft

        system.set_initial_state(geodetic_coordinates=geodetic_initial_pos)
        env.update(system)

        # TODO: improve initialization method
        alpha0 = 0.05
        beta0 = 0.001 * np.sign(turn_rate)

        ac._calculate_aerodynamics_2(TAS, alpha0, beta0, env)

        for control in ac.controls:
            if control not in initial_controls:
                raise ValueError("Control {} not given in initial_controls: {"
                                 "}".format(control, initial_controls))
            else:
                ac.controls[control] = initial_controls[control]

        controls_to_trim = list(ac.controls.keys() - exclude_controls)

        initial_guess = [alpha0, beta0]
        for control in controls_to_trim:
            initial_guess.append(initial_controls[control])

        lower_bounds = [-0.5, -0.25]  # Alpha and beta upper bounds.
        upper_bounds = [+0.5, +0.25]  # Alpha and beta lower bounds.
        for ii in controls_to_trim:
            lower_bounds.append(ac.control_limits[ii][0])
            upper_bounds.append(ac.control_limits[ii][1])
        bounds = (lower_bounds, upper_bounds)

        args = (system, ac, env, controls_to_trim, gamma, turn_rate)

        results = least_squares(trimming_cost_func,
                                x0=initial_guess,
                                args=args,
                                verbose=verbose,
                                bounds=bounds)
