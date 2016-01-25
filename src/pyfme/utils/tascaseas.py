# -*- coding: utf-8 -*-
"""
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

from math import sqrt

rho_0 = 1.225  # density at sea level (kg/m3)
p_0 = 101325  # pressure at sea level (Pa)
a_0 = 340.293990543  # sound speed at sea level (m/s)
gamma = 1.4  # heat capacity ratio
var = (gamma - 1)/gamma


def tas2eas(tas, rho):
    '''Given the True Airspeed, this function provides the Equivalent Airspeed.
        True Airspeed (TAS): is the speed of the aircraft relative to the
        airmass in which it is flying.

        Equivalent Airspeed (EAS): is the airspeed at sea level in the
        International Standard Atmosphere at which the dynamic pressure
        is the same as the dynamic pressure at the true airspeed (TAS) and
        altitude at which the aircraft is flying.

    Parameters
    ----------
    tas : float
        True Airspeed (TAS)
    rho : float
        Air density at flight level
    Returns
    -------
    eas : float
        Equivalent Airspeed (EAS)
    '''
    eas = tas * sqrt(rho / rho_0)

    return eas


def eas2tas(eas, rho):
    '''Given the Equivalent Airspeed, this function provides the True Airspeed.
        True Airspeed (TAS): is the speed of the aircraft relative to the
        airmass in which it is flying.

        Equivalent Airspeed (EAS): is the airspeed at sea level in the
        International Standard Atmosphere at which the dynamic pressure
        is the same as the dynamic pressure at the true airspeed (TAS) and
        altitude at which the aircraft is flying.

    Parameters
    ----------
    eas : float
        Equivalent Airspeed (EAS)
    rho : float
        Air density at flight level
    Returns
    -------
    tas : float
        True Airspeed (TAS)
    '''

    tas = eas / sqrt(rho / rho_0)

    return tas


def tas2cas(tas, p, rho):
    '''Given the True Airspeed, this function provides the Calibrated Airspeed.
        True Airspeed (TAS): is the speed of the aircraft relative to the
        airmass in which it is flying.

        Calibrated airspeed (CAS) is the speed shown by a conventional
        airspeed indicator after correction for instrument error and
        position error.

    Parameters
    ----------
    tas : float
        True Airspeed (TAS)
    p : float
        Air static pressure at flight level
    rho : float
        Air density at flight level
    Returns
    -------
    cas : float
        Calibrated Airspeed (CAS)
    '''

    a = sqrt(gamma * p / rho)

    cas = sqrt(2 * a_0 ** 2 / (gamma - 1) *
               ((((tas ** 2 * (gamma - 1) /
                  (2 * a ** 2) + 1) ** (1/var) - 1) *
                 (p / p_0) + 1) ** (var) - 1))

    return cas


def cas2tas(cas, p, rho):
    '''Given the Calibrated Airspeed, this function provides the True Airspeed.
        True Airspeed (TAS): is the speed of the aircraft relative to the
        airmass in which it is flying.

        Calibrated airspeed (CAS) is the speed shown by a conventional
        airspeed indicator after correction for instrument error and
        position error.

    Parameters
    ----------
    cas : float
        Calibrated Airspeed (CAS)
    p : float
        Air static pressur at flight level
    rho : float
        Air density at flight level
    Returns
    -------
    tas : float
        True Airspeed (TAS)
    '''

    a = sqrt(gamma * p / rho)

    tas = sqrt(2 * a ** 2 / (gamma - 1) *
               ((((cas ** 2 * (gamma - 1) /
                  (2 * a_0 ** 2) + 1) ** (1 / var) - 1) *
                 (p_0 / p) + 1) ** (var) - 1))

    return tas


def cas2eas(cas, p, rho):
    '''Given the Calibrated Airspeed, this function provides the Equivalent Airspeed.
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
        Calibrated Airspeed (CAS)
    p : float
        Air static pressur at flight level
    rho : float
        Air density at flight level
    Returns
    -------
    eas : float
        Equivalent Airspeed (EAS)
    '''

    tas = cas2tas(cas, p, rho)

    eas = tas2eas(tas, rho)

    return eas


def eas2cas(eas, p, rho):
    '''Given the Equivalent Airspeed, this function provides the Calibrated\
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
        Equivalent Airspeed (EAS)
    p : float
        Air static pressur at flight level
    rho : float
        Air density at flight level
    Returns
    -------
    cas : float
        Calibrated Airspeed (CAS)
    '''

    tas = eas2tas(eas, rho)

    cas = tas2cas(tas, p, rho)

    return cas


print(tas2cas(275, 22632.1, 0.36391861135917014))
