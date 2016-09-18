# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Constant variables
------------------
Sources:

[1] - COESA standard - U.S. Standard Atmosphere, 1976, U.S. Government Printing Office, Washington, D.C., 1976: 
      http://hdl.handle.net/2060/19770009539

[2] - "Introducción a la Ingenería Aeroespacial". Sebastián Franchini, Óscar López García. UPM

"""

# AIR CONSTANTS

GAMMA_AIR = 1.4  # Adiabatic index or ratio of specific heats (dry air at 20º C) - [1]
R_AIR = 287.05287  # Specific gas constant for dry air (J/(Kg·K))

# Air at sea level conditions h=0 (m)

RHO_0 = 1.225  # Density at sea level (kg/m3) - [1]
P_0 = 101325  # Pressure at sea level (Pa) - [1]
T_0 = 288.15 # Temperature at sea level (K) - [1]
SOUND_VEL_0 = 340.293990543  # Sound speed at sea level (m/s)


# EARTH CONSTANTS

GRAVITY = 9.80665  # Gravity of Ethe Earth (m/s^2) - [1]
STD_GRAVITATIONAL_PARAMETER = 3.986004418e14  # Standard Gravitational Parameter is the product of the gravitational
# constant G and the mass M of the body (m³/s²)
EARTH_MASS = 5.9722e24  # Mass of the Earth (kg)
GRAVITATIONAL_CONSTANT = 6.67384e11  # Gravitational constant (N·m²/kg²)
EARTH_MEAN_RADIUS = 6371000  # Mean radius of the Earth (m) - [2]


# CONVERSIONS

lbs2kg = 0.453592  # Pounds (lb) to kilograms (kg)
ft2m = 0.3048  # Feet (ft) to meters (m)
slug2kg = 14.5939  # Slug to kilograms (kg)
slugft2_2_kgm2 = 1.35581795  # Slug*feet^2 to kilograms*meters^2 (kg*m^2)