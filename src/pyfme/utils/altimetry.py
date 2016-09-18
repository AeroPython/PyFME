# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Altimetry
---------
Functions which transform geometric altitude into geopotential altitude, and
vice versa.
"""

from pyfme.models.constants import EARTH_MEAN_RADIUS

Re = EARTH_MEAN_RADIUS

def geopotential2geometric(h):
    """Geometric altiude above MSL (mean sea level)

    This function transforms geopotential altitude into geometric altitude.

    Parameters
    ----------
    h : float
        Geopotential altitude above MSL (mean sea level) (m)
    Returns
    -------
    z : float
        Geometric altitude above MSL (mean sea level) (m)

    See Also
    --------
    geopotential

    References
    ----------
    .. [1] International Organization for Standardization, Standard Atmosphere,
        ISO 2533:1975, 1975

    """

    z = Re * h / (Re - h)
    return z


def geometric2geopotential(z):
    """Geopotential altiude above MSL (mean sea level)

    This function transforms geometric altitude into geopotential altitude.

    Parameters
    ----------
    z : float
        Geometric altitude above MSL (mean sea level) (m)
    Returns
    -------
    h : float
        Geopotential altitude above MSL (mean sea level) (m)

    See Also
    --------
    geometric

    References
    ----------
    .. [1] International Organization for Standardization, Standard Atmosphere,
        ISO 2533:1975, 1975

    """
    
    h = Re * z / (Re + z)
    return h
