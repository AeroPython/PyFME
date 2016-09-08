"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Simulation class
----------------

"""
from abc import abstractmethod
import numpy as np


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
        self.aircraft = aircraft
        self.system = system
        self.environment = environment
        self._time_step = 0
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
        self.par_dict = {}

    def time_step(self, dt):
        """
        Performs a simulation time step.

        Parameters
        ----------
        dt : float
            Time step (s).
        """

        self.save_current_par_dict()
        self._time_step += 1
        self.system.propagate(self.aircraft, dt)
        self.environment.update(self.system)
        controls = self._get_current_controls(self._time_step)
        self.aircraft.update(controls, self.system, self.environment)
        self.aircraft.calculate_forces_and_moments()

    @abstractmethod
    def _get_current_controls(self, ii):
        return

    @abstractmethod
    def set_par_dict(self, par_list):
        return

    @abstractmethod
    def save_current_par_dict(self):
        return


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
