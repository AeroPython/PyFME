# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Anemometry related functions (to be completed)
----------------------------
Set of functions which allows to obtain the True Airspeed (TAS), the
Equivalent airspeed (EAS) or the Calibrated Airspeed (CAS) known one of the
others.

True Airspeed (TAS): is the speed of the aircraft relative to the
airmass in which it is flying.

Equivalent Airspeed (EAS): is the airspeed at sea level in the
International Standard Atmosphere at which the dynamic pressure
is the same as the dynamic pressure at the true airspeed (TAS) and
altitude at which the aircraft is flying.

Calibrated airspeed (CAS) is the speed shown by a conventional
airspeed indicator after correction for instrument error and
position error.
"""

from math import asin, atan, sqrt
from pyfme.models.constants import RHO_0, P_0, SOUND_VEL_0, GAMMA_AIR


rho_0 = RHO_0  # density at sea level (kg/m3)
p_0 = P_0  # pressure at sea level (Pa)
a_0 = SOUND_VEL_0  # sound speed at sea level (m/s)
gamma = GAMMA_AIR  # heat capacity ratio


def calculate_alpha_beta_TAS(u, v, w):
    """
    Calculate the angle of attack (AOA), angle of sideslip (AOS) and true air
    speed from the **aerodynamic velocity** in body coordinates.

    Parameters
    ----------
    u : float
        x-axis component of aerodynamic velocity. (m/s)
    v : float
        y-axis component of aerodynamic velocity. (m/s)
    w : float
        z-axis component of aerodynamic velocity. (m/s)

    Returns
    -------
    alpha : float
        Angle of attack (rad).
    betha : float
        Angle of sideslip (rad).
    TAS : float
        True Air Speed. (m/s)

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
        Dynamic pressure. (Pa)

    Notes
    -----
    $$ q_{inf} = 1/2  · rho · TAS² $$
    """

    return 0.5 * rho * TAS ** 2


def calculate_viscosity_Sutherland(T):
    """Calculates the viscosity of the air

    Parameters
    -----------
    T : float
        Temperature (K)

    Returns
    -----------
    visc : float
        viscosity of the air (kg/(m s))

    Notes
    -----------
    Acoording to [1] the limis for this function are:

    p < p_c =36 Atm (3.65 MPa)
    T < 2000 K

    According to [2] the limits for this function are:

    T < 550 K


    """

    visc_0 = 1.176*1e-5  # kg(m s)
    T_0 = 273.1  # K
    b = 0.4042  # nondimensional

    return visc_0 * (T / T_0)**(3 / 2) * ((1 + b)/((T / T_0) + b))


def tas2eas(tas, rho):
    """Given the True Airspeed, this function provides the Equivalent Airspeed.

    True Airspeed (TAS): is the speed of the aircraft relative to the
    airmass in which it is flying.

    Equivalent Airspeed (EAS): is the airspeed at sea level in the
    International Standard Atmosphere at which the dynamic pressure
    is the same as the dynamic pressure at the true airspeed (TAS) and
    altitude at which the aircraft is flying.

    Parameters
    ----------
    tas : float
        True Airspeed (TAS) (m/s)
    rho : float
        Air density at flight level (kg/m3)
    Returns
    -------
    eas : float
        Equivalent Airspeed (EAS) (m/s)
    """
    eas = tas * sqrt(rho / rho_0)

    return eas


def eas2tas(eas, rho):
    """Given the Equivalent Airspeed, this function provides the True Airspeed.

    True Airspeed (TAS): is the speed of the aircraft relative to the
    airmass in which it is flying.

    Equivalent Airspeed (EAS): is the airspeed at sea level in the
    International Standard Atmosphere at which the dynamic pressure
    is the same as the dynamic pressure at the true airspeed (TAS) and
    altitude at which the aircraft is flying.

    Parameters
    ----------
    eas : float
        Equivalent Airspeed (EAS) (m/s)
    rho : float
        Air density at flight level (kg/m3)
    Returns
    -------
    tas : float
        True Airspeed (TAS) (m/s)
    """

    tas = eas / sqrt(rho / rho_0)

    return tas


def tas2cas(tas, p, rho):
    """Given the True Airspeed, this function provides the Calibrated Airspeed.

    True Airspeed (TAS): is the speed of the aircraft relative to the
    airmass in which it is flying.

    Calibrated airspeed (CAS) is the speed shown by a conventional
    airspeed indicator after correction for instrument error and
    position error.

    Parameters
    ----------
    tas : float
        True Airspeed (TAS) (m/s)
    p : float
        Air static pressure at flight level (Pa)
    rho : float
        Air density at flight level (kg/m3)
    Returns
    -------
    cas : float
        Calibrated Airspeed (CAS) (m/s)
    """

    a = sqrt(gamma * p / rho)
    var = (gamma - 1) / gamma

    temp = (tas**2 * (gamma - 1) / (2 * a**2) + 1) ** (1/var)
    temp = (temp - 1) * (p / p_0)
    temp = (temp + 1) ** var - 1

    cas = sqrt(2 * a_0 ** 2 / (gamma - 1) * temp)

    return cas


def cas2tas(cas, p, rho):
    """Given the Calibrated Airspeed, this function provides the True Airspeed.

    True Airspeed (TAS): is the speed of the aircraft relative to the
    airmass in which it is flying.

    Calibrated airspeed (CAS) is the speed shown by a conventional
    airspeed indicator after correction for instrument error and
    position error.

    Parameters
    ----------
    cas : float
        Calibrated Airspeed (CAS) (m/s)
    p : float
        Air static pressure at flight level (Pa)
    rho : float
        Air density at flight level (kg/m3)
    Returns
    -------
    tas : float
        True Airspeed (TAS) (m/s)
    """

    a = sqrt(gamma * p / rho)
    var = (gamma - 1) / gamma

    temp = (cas**2 * (gamma - 1) / (2 * a_0**2) + 1) ** (1/var)
    temp = (temp - 1) * (p_0 / p)
    temp = (temp + 1) ** var - 1

    tas = sqrt(2 * a ** 2 / (gamma - 1) * temp)

    return tas


def cas2eas(cas, p, rho):
    """Given the Calibrated Airspeed, this function provides the Equivalent
    Airspeed.

    Calibrated airspeed (CAS) is the speed shown by a conventional
    airspeed indicator after correction for instrument error and
    position error.

    Equivalent Airspeed (EAS): is the airspeed at sea level in the
    International Standard Atmosphere at which the dynamic pressure
    is the same as the dynamic pressure at the true airspeed (TAS) and
    altitude at which the aircraft is flying.

    Parameters
    ----------
    cas : float
        Calibrated Airspeed (CAS) (m/s)
    p : float
        Air static pressure at flight level (Pa)
    rho : float
        Air density at flight level (kg/m3)
    Returns
    -------
    eas : float
        Equivalent Airspeed (EAS) (m/s)
    """

    tas = cas2tas(cas, p, rho)

    eas = tas2eas(tas, rho)

    return eas


def eas2cas(eas, p, rho):
    """Given the Equivalent Airspeed, this function provides the Calibrated
    Airspeed.

    Calibrated airspeed (CAS) is the speed shown by a conventional
    airspeed indicator after correction for instrument error and
    position error.

    Equivalent Airspeed (EAS): is the airspeed at sea level in the
    International Standard Atmosphere at which the dynamic pressure
    is the same as the dynamic pressure at the true airspeed (TAS) and
    altitude at which the aircraft is flying.

    Parameters
    ----------
    eas : float
        Equivalent Airspeed (EAS) (m/s)
    p : float
        Air static pressure at flight level (Pa)
    rho : float
        Air density at flight level (kg/m3)
    Returns
    -------
    cas : float
        Calibrated Airspeed (CAS) (m/s)
    """

    tas = eas2tas(eas, rho)

    cas = tas2cas(tas, p, rho)

    return cas


def stagnation_pressure(p, a, tas):
    """Given the static pressure, the sound velocity and the true air speed,
    it returns the stagnation pressure for a compressible flow.

    The stagnation pressure is the  is the static pressure a fluid retains when
    brought to rest isoentropically from Mach number M

    Subsonic case: Bernouilli's equation compressible form.

    Supersonic case: Due to the shock wave Bernouilli's equation is no longer
    applicable. Rayleigh Pitot tube formula is used.

    Parameters
    ----------
    tas : float
        True Airspeed (TAS) (m/s)
    p : float
        Air static pressure at flight level (Pa)
    a : float
        sound speed at flight level (m/s)
    Returns
    -------
    p_stagnation : float
        Stagnation pressure at flight level (Pa)
    References
    ----------
    .. [1] http://www.dept.aoe.vt.edu/~lutze/AOE3104/airspeed.pdf
    """

    var = (gamma - 1) / gamma
    M = tas/a

    if M < 1:
        p_stagnation = 1 + (gamma-1) * (M**2) / 2
        p_stagnation **= (1/var)
        p_stagnation *= p
    else:
        p_stagnation = (gamma+1)**2 * M**2
        p_stagnation /= (4*gamma*(M**2) - 2*(gamma-1))
        p_stagnation **= (1/var)
        p_stagnation *= (1 - gamma + 2*gamma*(M**2)) / (gamma+1)
        p_stagnation *= p

    return p_stagnation
