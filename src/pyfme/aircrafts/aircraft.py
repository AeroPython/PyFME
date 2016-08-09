"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Generic Aircraft
----------------

"""
from abc import abstractmethod
from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.optimize import least_squares

from pyfme.environment.environment import Environment
from pyfme.models.systems import System
from pyfme.utils.anemometry import tas2cas, tas2eas, calculate_alpha_beta_TAS
from pyfme.utils.trimmer import trimming_cost_func


class Aircraft(object):

    def __init__(self):
        # Mass & Inertia
        self.mass = 0  # kg
        self.inertia = 0  # kg·m²
        # Geometry
        self.Sw = 0  # m2
        self.chord = 0  # m
        self.span = 0  # m
        # Controls
        self.controls = {}
        self.control_limits = {}
        # Coefficients
        # Aero
        self.CL, self.CD, self.Cm = 0, 0, 0
        self.CY, self.Cl, self.Cn = 0, 0, 0
        # Thrust
        self.Ct = 0
        # Forces & moments
        self.gravity_force = np.zeros(3)
        self.total_forces = np.zeros(3)
        self.total_moments = np.zeros(3)
        self.load_factor = 0

        # Velocities
        self.TAS = 0  # True Air Speed.
        self.CAS = 0  # Calibrated Air Speed.
        self.EAS = 0  # Equivalent Air Speed.
        self.Mach = 0  # Mach number

        self.q_inf = 0  # Dynamic pressure at infty (Pa)

        # Angles
        self.alpha = 0  # Angle of attack (AOA).
        self.beta = 0  # Angle of sideslip (AOS).
        # Not present in this model:
        self.Dalpha_Dt = 0  # Rate of change of AOA.
        self.Dbeta_Dt = 0  # Rate of change of AOS.

    @property
    def Ixx(self):
        return self.inertia[0, 0]

    @property
    def Iyy(self):
        return self.inertia[1, 1]

    @property
    def Izz(self):
        return self.inertia[2, 2]

    def update(self, controls, system, environment):

        if self.controls_inside_range(controls):
            self.controls = controls

        # Velocity relative to air: aerodynamic velocity.
        aero_vel = system.vel_body - environment.body_wind

        alpha, beta, self.TAS = calculate_alpha_beta_TAS(
            u=aero_vel[0], v=aero_vel[1], w=aero_vel[2])

        # Calculate alpha and beta rate of change using finite differences.
        self.Dalpha_Dt = (alpha - self.alpha) / system.dt
        self.alpha = alpha
        self.Dbeta_Dt = (beta - self.beta) / system.dt
        self.alpha = alpha
        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a
        self.q_inf = 0.5 * environment.rho * self.TAS ** 2
        # Gravity force
        self.gravity_force = environment.gravity_vector * self.mass

    @abstractmethod
    def calculate_forces_and_moments(self):
        pass

    @abstractmethod
    def controls_inside_range(self, controls):
        for con, val in controls.items():
            limits = self.control_limits[con]
            is_correct = limits[0] <= val <= limits[1]
            if not is_correct:
                break
        return is_correct

    def steady_state_flight_trim(self, system, env, controls, TAS, gamma=0.,
                                 turn_rate=0., controls2trim=None,
                                 verbose=0.):
        # TODO: write docstring again
        """Finds a combination of values of the state and control variables that
        correspond to a steady-state flight condition. Steady-state aircraft flight
        can be defined as a condition in which all of the motion variables are
        constant or zero. That is, the linear and angular velocity components are
        constant (or zero), and all acceleration components are zero.

        Parameters
        ----------
        aircraft : aircraft class
            Aircraft class with methods get_forces, get_moments
        h : float
            Geopotential altitude for ISA (m).
        TAS : float
            True Air Speed (m/s).
        gamma : float, optional
            Flight path angle (rad).
        turn_rate : float, optional
            Turn rate, d(psi)/dt (rad/s).

        Returns
        -------
        lin_vel : float array
            [u, v, w] air linear velocity body-axes (m/s).
        ang_vel : float array
            [p, q, r] air angular velocity body-axes (m/s).
        theta : float
            Pitch angle (rad).
        phi : float
            Bank angle (rad).
        alpha : float
            Angle of attack (rad).
        beta : float
            Sideslip angle (rad).
        control_vector : array_like
            [delta_e, delta_ail, delta_r, delta_t].

        Notes
        -----
        See section 3.4 in [1] for the algorithm description.
        See section 2.5 in [1] for the definition of steady-state flight condition.

        References
        ----------
        .. [1] Stevens, BL and Lewis, FL, "Aircraft Control and Simulation",
            Wiley-lnterscience.
        """
        trim_system = deepcopy(system)
        trim_env = deepcopy(env)

        self.TAS = TAS
        self.Mach = self.TAS / env.a
        self.q_inf = 0.5 * trim_env.rho * self.TAS ** 2

        # Update environment
        trim_env.update(trim_system)

        if controls2trim is None:
            controls2trim = list(controls.keys())

        # TODO: try to look for a good inizialization method for alpha & beta
        initial_guess = [0.05,  # alpha
                         0.001 * np.sign(turn_rate)]  # beta
        for control in controls2trim:
            initial_guess.append(controls[control])

        args = (trim_system, self, trim_env, controls2trim, gamma, turn_rate)

        lower_bounds = [-0.5, -0.25]  # Alpha and beta upper bounds.
        upper_bounds = [+0.5, +0.25]  # Alpha and beta lower bounds.
        for ii in controls2trim:
            lower_bounds.append(self.control_limits[ii][0])
            upper_bounds.append(self.control_limits[ii][1])
        bounds = (lower_bounds, upper_bounds)

        results = least_squares(trimming_cost_func, x0=initial_guess, args=args,
                                verbose=verbose, bounds=bounds)

        trimmed_params = results['x']
        fun = results['fun']
        cost = results['cost']

        if cost > 1e-7 or any(abs(fun) > 1e-3):
            warn("Trim process did not converge", RuntimeWarning)

        trim_system.set_initial_state_vector()

        outputs = {'alpha': self.alpha, 'beta': self.beta,
                   'u': trim_system.u, 'v': trim_system.v, 'w': trim_system.w,
                   'p': trim_system.p, 'q': trim_system.q, 'r': trim_system.r,
                   'psi': trim_system.psi, 'theta': trim_system.theta,
                   'phi': trim_system.phi}

        trimmed_controls = controls
        for ii, jj in enumerate(range(2, len(trimmed_params))):
            trimmed_controls[controls2trim[ii]] = trimmed_params[jj]

        return trimmed_controls, trim_system, outputs
