# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Cessna 310
----------

"""
import numpy as np

from pyfme.aircrafts.aircraft import Aircraft
from pyfme.environment.environment import Environment
from pyfme.models.systems import System
from pyfme.utils.anemometry import calculate_dynamic_pressure
from pyfme.utils.coordinates import hor2body
from pyfme.models.constants import ft2m, slug2kg, slugft2_2_kgm2, lbs2kg


class Cessna310(Aircraft):
    """
    Cessna 310

    The Cessna 310 is an American six-seat, low-wing, twin-engined monoplane
    that was produced by Cessna between 1954 and 1980. It was the first
    twin-engined aircraft that Cessna put into production after World War II.

    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """

    def __init__(self):

        self.mass = 4600 * lbs2kg   # kg
        self.inertia = np.diag([8884, 1939, 11001]) * slugft2_2_kgm2  # kg·m²
        # Ixz_b = 0 * slugft2_2_kgm2  # Kg * m2

        self.Sw = 175 * ft2m**2  # m2
        self.chord = 4.79 * ft2m  # m
        self.span = 36.9 * ft2m  # m

        # |CL|   |CL_0  CL_a  CL_de  CL_dih| |  1     |
        # |CD| = |CD_0  CD_a  0      0     | | alpha  |
        # |Cm|   |Cm_0  Cm_a  Cm_de  Cm_dih| |delta_e |
        #                                    |delta_ih|
        self._long_inputs = np.zeros(4)
        self._long_inputs[0] = 1.0
        self._long_coef_matrix = np.array([[0.288,   4.58,  0.81, 0],
                                           [0.029,  0.160,     0, 0],
                                           [ 0.07, -0.137, -2.26, 0]])

        self.CL = 0
        self.CD = 0
        self.Cm = 0

        # |CY|   |CY_b  CY_da  CY_dr| |beta   |
        # |Cl| = |Cl_b  Cl_da  Cl_dr| |delta_a|
        # |Cn|   |Cn_b  Cn_da  Cn_dr| |delta_r|
        self._lat_inputs = np.zeros(3)
        self._lat_coef_matrix = np.array([[-0.698 ,       0,   0.230],
                                          [-0.1096,   0.172,  0.0192],
                                          [ 0.1444, -0.0168, -0.1152]])
        self.CY = 0
        self.Cl = 0
        self.Cn = 0

        self.control_names = ['delta_elevator', 'hor_tail_incidence',
                              'delta_aileron', 'delta_rudder', 'delta_t']

        self.forces = np.zeros(3)
        self.moments = np.zeros(3)
        """
    CY_b is the side force stability derivative with respect to the
        angle of sideslip
    CY_da is the side force stability derivative with respect to the
         aileron deflection
    CY_dr is the side force stability derivative with respect to the
         rudder deflection

    Cl_b is the rolling moment stability derivative with respect to
        angle of sideslip
    Cl_da is the rolling moment stability derivative with respect to
        the aileron deflection
    Cl_dr is the rolling moment stability derivative with respect to
        the rudder deflection

    Cn_b is the yawing moment stability derivative with respect to the
        angle of sideslip
    Cn_da is the yawing moment stability derivative with respect to the
        aileron deflection
    Cn_dr is the yawing moment stability derivative with respect to the
        rudder deflection

    CL_0 is the lift coefficient evaluated at the initial condition
    CL_a is the lift stability derivative with respect to the angle of attack
    CL_de is the lift stability derivative with respect to the elevator
         deflection
    CL_dih is the lift stability derivative with respect to the stabilator
         deflection
    CD_0 is the drag coefficient evaluated at the initial condition
    CD_a is the drag stability derivative with respect to the angle of attack
    Cm_0 is the pitching moment coefficient evaluated at the condition
        (alpha0 = deltaE = deltaih = 0º)
    Cm_a is the pitching moment stability derivative with respect to the angle
        of attack
    Cm_de is the pitching moment stability derivative with respect to the
        elevator deflection
    Cm_dih is the pitching moment stability derivative with respect to the
         stabilator deflection
    """

    def _set_aero_lon_forces_moments(self, system: System, controls: dict):

        self._long_inputs[1] = system.alpha
        self._long_inputs[2] = controls['delta_elevator']
        self._long_inputs[3] = controls['hor_tail_incidence']

        self.CL, self.CD, self.Cm = self._long_coef_matrix @ self._long_inputs

    def _set_aero_lat_forces_moments(self, system: System, controls: dict):
        self._lat_inputs[0] = system.beta
        self._lat_inputs[1] = controls['delta_aileron']
        self._lat_inputs[2] = controls['delta_rudder']

        self.CY, self.Cl, self.Cn = self._lat_coef_matrix @ self._lat_inputs

    def _set_thrust_forces_moments(self, system: System, controls: dict):

        delta_t = controls['delta_t']
        self.Ct = 0.031 * delta_t

    def get_forces_and_moments(self, system: System, controls: dict,
                               env: Environment):

        q = system.q_inf
        Sw = self.Sw
        c = self.chord
        b = self.span
        self.check_control_limits()
        self._set_aero_lon_forces_moments(system, controls)
        self._set_aero_lat_forces_moments(system, controls)
        self._set_thrust_forces_moments(system, controls)

        L = q * Sw * self.CL
        D = q * Sw * self.CD
        Y = q * Sw * self.CY
        l = q * Sw * b * self.Cl
        m = q * Sw * c * self.Cm
        n = q * Sw * b * self.Cn

        Ft = q * Sw * self.Ct
        Fg = self.mass * env.gravity_vector
        Fa = np.array([-D, Y, -L])

        self.forces = Ft + Fg + Fa
        self.moments = np.array([l, m, n])
        return self.forces, self.moments
