# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Cessna 310
----------

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
import numpy as np

from pyfme.aircrafts.aircraft import Aircraft
from pyfme.models.constants import ft2m, slugft2_2_kgm2, lbs2kg
from pyfme.utils.coordinates import wind2body


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

        # Mass & Inertia
        self.mass = 4600 * lbs2kg   # kg
        self.inertia = np.diag([8884, 1939, 11001]) * slugft2_2_kgm2  # kg·m²
        # Ixz_b = 0 * slugft2_2_kgm2  # Kg * m2

        # Geometry
        self.Sw = 175 * ft2m**2  # m2
        self.chord = 4.79 * ft2m  # m
        self.span = 36.9 * ft2m  # m

        # Aerodynamic Data (Linearized)
        # |CL|   |CL_0  CL_a  CL_de  CL_dih| |  1     |
        # |CD| = |CD_0  CD_a  0      0     | | alpha  |
        # |Cm|   |Cm_0  Cm_a  Cm_de  Cm_dih| |delta_e |
        #                                    |delta_ih|
        #
        # |CY|   |CY_b  CY_da  CY_dr| |beta   |
        # |Cl| = |Cl_b  Cl_da  Cl_dr| |delta_a|
        # |Cn|   |Cn_b  Cn_da  Cn_dr| |delta_r|
        self._long_coef_matrix = np.array([[0.288,   4.58,  0.81, 0],
                                           [0.029,  0.160,     0, 0],
                                           [ 0.07, -0.137, -2.26, 0]])

        self._lat_coef_matrix = np.array([[-0.698 ,       0,   0.230],
                                          [-0.1096,   0.172,  0.0192],
                                          [ 0.1444, -0.0168, -0.1152]])
        self._long_inputs = np.zeros(4)
        self._long_inputs[0] = 1.0

        self._lat_inputs = np.zeros(3)

        # CONTROLS
        self.controls = {'delta_elevator': 0,
                         'hor_tail_incidence': 0,
                         'delta_aileron': 0,
                         'delta_rudder': 0,
                         'delta_t': 0}

        # FIXME: these limits are not real
        self.control_limits = {'delta_elevator': (-1, 1),  # rad
                               'hor_tail_incidence': (-1, 1),  # rad
                               'delta_aileron': (-1, 1),  # rad
                               'delta_rudder': (-1, 1),  # rad
                               'delta_t': (0, 1)}  # non dimensional

        # Coefficients
        # Aero
        self.CL, self.CD, self.Cm = 0, 0, 0
        self.CY, self.Cl, self.Cn = 0, 0, 0
        # Thrust
        self.Ct = 0

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

    def _calculate_aero_lon_forces_moments_coeffs(self):

        self._long_inputs[1] = self.alpha
        self._long_inputs[2] = self.controls['delta_elevator']
        self._long_inputs[3] = self.controls['hor_tail_incidence']

        self.CL, self.CD, self.Cm = self._long_coef_matrix @ self._long_inputs

    def _calculate_aero_lat_forces_moments_coeffs(self):
        self._lat_inputs[0] = self.beta
        self._lat_inputs[1] = self.controls['delta_aileron']
        self._lat_inputs[2] = self.controls['delta_rudder']

        self.CY, self.Cl, self.Cn = self._lat_coef_matrix @ self._lat_inputs

    def _calculate_aero_forces_moments(self):
        q = self.q_inf
        Sw = self.Sw
        c = self.chord
        b = self.span
        self._calculate_aero_lon_forces_moments_coeffs()
        self._calculate_aero_lat_forces_moments_coeffs()
        L = q * Sw * self.CL
        D = q * Sw * self.CD
        Y = q * Sw * self.CY
        l = q * Sw * b * self.Cl
        m = q * Sw * c * self.Cm
        n = q * Sw * b * self.Cn
        return L, D, Y, l, m , n

    def _calculate_thrust_forces_moments(self):
        q = self.q_inf
        Sw = self.Sw
        self.Ct = 0.031 * self.controls['delta_t']
        Ft = np.array([q * Sw * self.Ct, 0, 0])
        return Ft


    def calculate_forces_and_moments(self):

        Ft = self._calculate_thrust_forces_moments()
        L, D, Y, l, m, n = self._calculate_aero_forces_moments()
        Fg = self.gravity_force
        # FIXME: is it necessary to use wind2body conversion?
        Fa = np.array([-D, Y, -L])

        self.total_forces = 10 * Ft + Fg + Fa
        self.total_moments = np.array([l, m, n])
        return self.total_forces, self.total_moments
