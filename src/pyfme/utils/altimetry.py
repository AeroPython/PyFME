"""
Functions which transform geometric altitude into geopotential altitude, and
vice versa.

"""


def geometric(h):
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

    Re = 6371000  # Mean radius Earth (m)
    z = Re * h / (Re - h)
    return z


def geopotential(z):
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

    Re = 6371000  # Mean radius Earth (m)
    h = Re * z / (Re + z)
    return h
