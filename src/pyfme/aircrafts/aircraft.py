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
        self.propeller_radius = 0  # m
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

        # Angular velocities
        self.p = 0  # rad/s
        self.q = 0  # rad/s
        self.r = 0  # rad/s

        # Angles
        self.alpha = 0  # Angle of attack (AOA).
        self.beta = 0  # Angle of sideslip (AOS).
        self.alpha_dot = 0  # Rate of change of AOA.
        #NOT PRESENT self.Dbeta_Dt = 0  # Rate of change of AOS.
        # Environment
        self.rho = 0  # kg/m3

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

        # If a control is not given, the previous value is assigned.
        for control_name, control_value in controls.items():
            limits = self.control_limits[control_name]
            if limits[0] <= control_value <= limits[1]:
                self.controls[control_name] = control_value
            else:
                # TODO: maybe raise a warning and assign max deflection
                msg = "Control {} out of range ({} when max={} and min={" \
                      "}".format(control_name, limits[1], limits[0])
                raise ValueError(msg)

        # Velocity relative to air: aerodynamic velocity.
        aero_vel = system.vel_body - environment.body_wind

        self.alpha, self.beta, self.TAS = calculate_alpha_beta_TAS(
            u=aero_vel[0], v=aero_vel[1], w=aero_vel[2])

        self.p, self.q, self.r = system.vel_ang

        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a
        self.q_inf = 0.5 * environment.rho * self.TAS ** 2

        # Setting environment
        self.rho = environment.rho

        # Gravity force
        self.gravity_force = environment.gravity_vector * self.mass
