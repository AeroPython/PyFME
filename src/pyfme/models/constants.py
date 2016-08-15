"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Constant variables
------------------

"""
# AIR
GAMMA_AIR = 1.401  # Adiabatic index or ratio of specific heats (dry air at
                   # 20º C)
R_AIR = 287.05287  # J/(Kg·K)
# Sea level conditions
RHO_0 = 1.225  # density at sea level (kg/m3)
P_0 = 101325  # pressure at sea level (Pa)
SOUND_VEL_0 = 340.293990543  # sound speed at sea level (m/s)

# GRAVITY of EARTH
GRAVITY = 9.80665  # m/s^2
# Standard Gravitational Parameter is the product of the gravitational
# constant G and the mass M of the body.
STD_GRAVITATIONAL_PARAMETER = 3.986004418e14  # m³/s²
EARTH_MASS = 5.9722e24  # kg
GRAVITATIONAL_CONSTANT = 6.67384e11 # N·m²/kg²

# Conversions
lbs2kg = 0.453592
ft2m = 0.3048
slug2kg = 14.5939
slugft2_2_kgm2 = 1.35581795