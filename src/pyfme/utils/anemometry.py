# -*- coding: utf-8 -*-
"""
Anemometry related functions
"""

from math import asin, atan, sqrt


def calculate_alpha_beta_TAS(u, v, w):
    """
    Calculate the angle of attack (AOA), angle of sideslip (AOS) and true air
    speed from the **aerodynamic velocity** in body coordinates.

    Parameters
    ----------
    u : float
        x-axis component of aerodynamic velocity.
    v : float
        y-axis component of aerodynamic velocity.
    w : float
        z-axis component of aerodynamic velocity.

    Returns
    -------
    alpha : float
        Angle of attack.
    betha : float
        Angle of sideslip.
    TAS : float
        True Air Speed.

    Notes
    -----
    See [1] or [2] for frame of reference definition.
    See [3] for formula derivation.

    $$ TAS = sqrt(u^2 + v^2 + w^2)$$

    $$ alpha = \atan(w / u) $$
    $$ beta = \asin(v / TAS) $$

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight," Courier Corporation,
        pp. 104-120, 2012.
    .. [2] Gómez Tierno, M.A. et al, "Mecánica del Vuelo," Garceta, pp. 1-12,
        2012.
    .. [3] Stevens, BL and Lewis, FL, "Aircraft Control and Simulation",
        Wiley-lnterscience, pp. 64, 1992.
    """

    TAS = sqrt(u ** 2 + v ** 2 + w ** 2)

    alpha = atan(w / u)
    beta = asin(v / TAS)

    return alpha, beta, TAS


def calculate_mach(TAS, T, gamma=1.401, R_a=287.05287):
    """Calculates the Mach number given the True Air Speed (TAS) and the
    temperature.

    Parameters
    ----------
    TAS : float
        True Air Speed (m/s).
    T : float
        Temperature (K).
    gamma : float, optional
        Adiabatic index or ratio of specific heats. Default value corresponds
        to dry air at 20º C.
    R_a : float, optional.
        Ideal gas constant J/(Kg·K). Default value corresponds to idel air.

    Returns
    -------
    mach : float
        Mach number

    Notes
    -----
    $$ a = \sqrt(\gamma R_g T) $$

    Adiabatic index, gamma = 1.401 from -40 ºC to 21 ºC. Variations outside
    that range are neglible, a priori.
    """

    return TAS / sqrt(gamma * R_a * T)


def calculate_dynamic_pressure(rho, TAS):
    """Calculates the dynamic pressure.

    Parameters
    ----------
    rho : float
        Air density (kg/m³).
    TAS : float
        True Air Speed (m/s).

    Returns
    -------
    q_inf : float
        Dynamic pressure.

    Notes
    -----
    $$ q_{inf} = 1/2  · rho · TAS² $$
    """

    return 0.5 * rho * TAS ** 2
