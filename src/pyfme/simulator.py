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

        self.vars_to_save = {'h': 'system.height'}
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


class BatchSimulation(Simulation):

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
        super().__init__(aircraft, system, environment)
        self.time = None
        self.aircraft_controls = {}

    def set_controls(self, time, controls):
        """Set the time history of controls and the corresponding times.

        Parameters
        ----------
        time : array_like
            Time history for the simulation.
        controls : array_like
            Controls for the given time array.
        """

        # check time dimensions
        if time.ndim != 1:
            raise ValueError('Time must be unidimensional')
        tsize = time.size
        # check dimensions
        for c in controls:
            if controls[c].size != tsize:
                msg = 'Control {} size ({}) does not match time size ({' \
                      '})'.fromat(c, c.size, tsize)
                raise ValueError(msg)
        self.time = time
        self.aircraft_controls = controls

    def _get_current_controls(self, ii):
        """Returns controls at time step ii.

        Parameters
        ----------
        ii

        Returns
        -------

        """
        return {control: self.aircraft_controls[control][ii] for control in
                self.aircraft_controls.keys()}

    def run_simulation(self):
        """
        Run simulation for the times in self.time.
        """
        if self.time is None:
            raise ValueError("Time and controls for the simulation must be "
                             "set with `set_controls()`")

        for ii, t in enumerate(self.time[1:]):
            dt = t - self.time[ii]
            self.time_step(dt)
        # Save last time step
        self.save_current_par_dict()

    def set_par_dict(self, par_list):
        """
        Set parameters to be saved

        Parameters
        ----------
        par_list : list
            List with parameter names.
        """
        if self.time is None:
            msg = "Set controls with BatchSimulation.set_controls before " \
                  "setting the par_dict"
            raise RuntimeError(msg)

        for par_name in par_list:
            if par_name in self.PAR_KEYS:
                self.par_dict[par_name] = np.empty_like(self.time)
            else:
                msg = "{} not found in PAR_KEYS".format(par_name)
                raise RuntimeWarning(msg)

    def save_current_par_dict(self):
        self.PAR_KEYS = {'T': self.environment.T,  # env
                         'pressure': self.environment.p,
                         'rho': self.environment.rho,
                         'a': self.environment.a,
                         'TAS': self.aircraft.TAS,  # aircraft
                         'Mach': self.aircraft.Mach,
                         'q_inf': self.aircraft.q_inf,
                         'alpha': self.aircraft.alpha,
                         'beta': self.aircraft.beta,
                         'x_earth': self.system.x_earth,  # system
                         'y_earth': self.system.y_earth,
                         'z_earth': self.system.z_earth,
                         'psi': self.system.psi,
                         'theta': self.system.theta,
                         'phi': self.system.phi,
                         'u': self.system.u,
                         'v': self.system.v,
                         'w': self.system.w,
                         'v_north': self.system.v_north,
                         'v_east': self.system.v_east,
                         'v_down': self.system.v_down,
                         'p': self.system.p,
                         'q': self.system.q,
                         'r': self.system.r,
                         'height': self.system.height,
                         'F_xb': self.aircraft.total_forces[0],
                         'F_yb': self.aircraft.total_forces[1],
                         'F_zb': self.aircraft.total_forces[2],
                         'M_xb': self.aircraft.total_moments[0],
                         'M_yb': self.aircraft.total_moments[1],
                         'M_zb': self.aircraft.total_moments[2]
                         }
        for par_name, par_values in self.par_dict.items():
            par_values[self._time_step] = self.PAR_KEYS[par_name]


class RealTimeSimulation(Simulation):

    def __init__(self, aircraft, system, environment):
        raise NotImplementedError()
        super(RealTimeSimulation, self).__init__(aircraft, system, environment)
        # TODO:...

    def _get_current_controls(self, ii):
        # Joystick reading
        raise NotImplementedError

    def set_par_dict(self, par_list):

        for par_name in par_list:
            if par_name in self.PAR_KEYS:
                self.par_dict[par_name] = []
            else:
                msg = "{} not found in PAR_KEYS".format(par_name)
                raise RuntimeWarning(msg)

    def save_current_par_dict(self):
        self.PAR_KEYS = {'T': self.environment.T,  # env
                         'pressure': self.environment.p,
                         'rho': self.environment.rho,
                         'a': self.environment.a,
                         'TAS': self.aircraft.TAS,  # aircraft
                         'Mach': self.aircraft.Mach,
                         'q_inf': self.aircraft.q_inf,
                         'alpha': self.aircraft.alpha,
                         'beta': self.aircraft.beta,
                         'x_earth': self.system.x_earth,  # system
                         'y_earth': self.system.y_earth,
                         'z_earth': self.system.z_earth,
                         'psi': self.system.psi,
                         'theta': self.system.theta,
                         'phi': self.system.phi,
                         'u': self.system.u,
                         'v': self.system.v,
                         'w': self.system.w,
                         'v_north': self.system.v_north,
                         'v_east': self.system.v_east,
                         'v_down': self.system.v_down,
                         'p': self.system.p,
                         'q': self.system.q,
                         'r': self.system.r,
                         'height': self.system.height
                         }
        for par_name, par_values in self.par_dict:
            par_values.append(self.PAR_KEYS[par_name])
