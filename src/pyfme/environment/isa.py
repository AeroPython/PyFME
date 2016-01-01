# coding: utf-8
"""ISA functions. Implementation based on:

.. [1] U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
        Washington, D.C., 1976

 From: https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere

               Geopotential
               altitude     Static       Standard     Temperature
    Subscript  above MSL    Pressure     Temperature  Lapse Rate
    -------------------------------------------------------------
                (m)	       (pascals)     (K)         (K/m)

    0	           0	       101325	    288.15	     -0.0065
    1	          11000	       22632.1	    216.65	      0
    2	          20000	       5474.89	    216.65	      0.001
    3	          32000	       868.019	    228.65	      0.0028
    4	          47000	       110.906	    270.65	      0
    5	          51000	       66.9389	    270.65	     -0.0028
    6	          71000	       3.95642	    214.65	     -0.002

"""

from math import exp

# Constants
R_a = 287.05287  # J/(Kg·K)
g0 = 9.80665  # m/s^2

# Layer constants
h0 = (0, 11000, 20000, 32000, 47000, 51000, 71000, 84500)  # m
T0_layers = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65)  # K
p0_layers = (101325.0, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642)  # Pa
alpha_layers = (-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002)  # K / m


def atm(h):
    """ISA 1976 Standard atmosphere temperature, pressure and density.

    Parameters
    ----------
    h : float
        Geopotential altitude (m). h values must range from 0 to 84500 m.

    Returns
    -------
    T : float
        Temperature (K).
    p : float
        Pressure (Pa).
    rho : float
        Density (kg/m³)

    Raises
    ------
    ValueError
        If the value of the altitude is outside the defined layers.

    Notes
    -----
    Check layers and reference values in [2].

    References
    ----------
    .. [1] U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
        Washington, D.C., 1976
    .. [2] https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere

    """

    if h < 0.0:
        raise ValueError("Altitude cannot be less than 0 m.")

    elif h0[0] <= h < h0[1]:  # Troposphere
        T0 = T0_layers[0]
        p0 = p0_layers[0]
        alpha = alpha_layers[0]

        T = T0 + alpha * h
        p = p0 * (T0 / (T0 + alpha * h)) ** (g0 / (R_a * alpha))

    elif h0[1] <= h < h0[2]:  # Tropopause
        T0 = T0_layers[1]
        p0 = p0_layers[1]
        alpha = alpha_layers[1]
        h_diff = h - h0[1]

        T = T0
        p = p0 * exp(-g0 * h_diff / (R_a * T0))

    elif h0[2] <= h < h0[3]:  # Stratosphere 1
        T0 = T0_layers[2]
        p0 = p0_layers[2]
        alpha = alpha_layers[2]
        h_diff = h - h0[2]

        T = T0 + alpha * h_diff
        p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

    elif h0[3] <= h < h0[4]:  # Stratosphere 2
        T0 = T0_layers[3]
        p0 = p0_layers[3]
        alpha = alpha_layers[3]
        h_diff = h - h0[3]

        T = T0 + alpha * h_diff
        p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

    elif h0[4] <= h < h0[5]:  # Stratopause
        T0 = T0_layers[4]
        p0 = p0_layers[4]
        alpha = alpha_layers[4]
        h_diff = h - h0[4]

        T = T0
        p = p0 * exp(-g0 * h_diff / (R_a * T0))

    elif h0[5] <= h < h0[6]:  # Mesosphere 1
        T0 = T0_layers[5]
        p0 = p0_layers[5]
        alpha = alpha_layers[5]
        h_diff = h - h0[5]

        T = T0 + alpha * h_diff
        p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

    elif h0[6] <= h <= h0[7]:  # Mesosphere 2
        T0 = T0_layers[6]
        p0 = p0_layers[6]
        alpha = alpha_layers[6]
        h_diff = h - h0[6]

        T = T0 + alpha * h_diff
        p = p0 * (T0 / (T0 + alpha * h_diff)) ** (g0 / (R_a * alpha))

    else:
        raise ValueError("Altitude cannot be greater than {} m.".format(h0[7]))

    rho = p / (R_a * T)

    return T, p, rho
