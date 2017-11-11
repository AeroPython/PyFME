"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Generic Aircraft
----------------

"""
from abc import abstractmethod

import numpy as np

from pyfme.utils.anemometry import tas2cas, tas2eas, calculate_alpha_beta_TAS


class Aircraft(object):

    def __init__(self):
        # Mass & Inertia
        self.mass = 0  # kg
        self.inertia = np.zeros((3, 3))  # kg·m²

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
        self.total_forces = np.zeros(3)
        self.total_moments = np.zeros(3)

        # Velocities
        self.TAS = 0  # True Air Speed.
        self.CAS = 0  # Calibrated Air Speed.
        self.EAS = 0  # Equivalent Air Speed.
        self.Mach = 0  # Mach number
        self.q_inf = 0  # Dynamic pressure at infty (Pa)

        # Angles
        self.alpha = 0  # Angle of attack (AOA).
        self.beta = 0  # Angle of sideslip (AOS).
        self.alpha_dot = 0  # Rate of change of AOA.

    @property
    def Ixx(self):
        return self.inertia[0, 0]

    @property
    def Iyy(self):
        return self.inertia[1, 1]

    @property
    def Izz(self):
        return self.inertia[2, 2]

    @property
    def Fx(self):
        return self.total_forces[0]

    @property
    def Fy(self):
        return self.total_forces[1]

    @property
    def Fz(self):
        return self.total_forces[2]

    @property
    def Mx(self):
        return self.total_moments[0]

    @property
    def My(self):
        return self.total_moments[1]

    @property
    def Mz(self):
        return self.total_moments[2]

    def _set_current_controls(self, controls):

        # If a control is not given, the previous value is assigned.
        for control_name, control_value in controls.items():
            limits = self.control_limits[control_name]
            if limits[0] <= control_value <= limits[1]:
                self.controls[control_name] = control_value
            else:
                # TODO: maybe raise a warning and assign max deflection
                msg = (
                    f"Control {control_name} out of range ({control_value} "
                    f"when min={limits[0]} and max={limits[1]})"
                )
                raise ValueError(msg)

    def _calculate_aerodynamics(self, state, environment):

        # Velocity relative to air: aerodynamic velocity.
        aero_vel = state.velocity.vel_body - environment.body_wind

        self.alpha, self.beta, self.TAS = calculate_alpha_beta_TAS(
            u=aero_vel[0], v=aero_vel[1], w=aero_vel[2]
        )

        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a
        self.q_inf = 0.5 * environment.rho * self.TAS ** 2

    def _calculate_aerodynamics_2(self, TAS, alpha, beta, environment):

        self.alpha, self.beta, self.TAS = alpha, beta, TAS

        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a
        self.q_inf = 0.5 * environment.rho * self.TAS ** 2

    @abstractmethod
    def calculate_forces_and_moments(self, state, environment, controls):

        self._set_current_controls(controls)
        self._calculate_aerodynamics(state, environment)
