"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Position
--------
Aircraft position class prepared to represent the aircraft position in Earth
axis, geodetic coordinates and geocentric coordinates independently of the
dynamic system used
"""
from abc import abstractmethod

import numpy as np

from pyfme.models.constants import EARTH_MEAN_RADIUS


class Position:
    """Position

    Attributes
    ----------

    geodetic_coordinates : ndarray, shape(3)
        (lat [rad], lon [rad], height [m])
    lat
    lon
    height
    geocentric_coordinates : ndarray, shape(3)
        (x_geo [m], y_geo [m], z_geo [m])
    x_geo
    y_geo
    z_geo
    earth_coordinates : ndarray, shape(3)
        (x_earth [m], y_earth [m], z_earth [m])
    x_earth
    y_earth
    z_earth
    """

    def __init__(self, geodetic, geocentric, earth):
        # Geodetic coordinates: (geodetic lat, lon, height above ellipsoid)
        self._geodetic_coordinates = np.asarray(geodetic)  # rad
        # Geocentric coordinates (rotating with Earth): (x_geo, y_geo, z_geo)
        self._geocentric_coordinates = np.asarray(geocentric)  # m
        # Earth coordinates (x_earth, y_earth, z_earth)
        self._earth_coordinates = np.asarray(earth)  # m

    @abstractmethod
    def update(self, coords):
        raise NotImplementedError

    @property
    def geocentric_coordinates(self):
        return self._geocentric_coordinates

    @property
    def x_geo(self):
        return self._geocentric_coordinates[0]

    @property
    def y_geo(self):
        return self._geocentric_coordinates[1]

    @property
    def z_geo(self):
        return self._geocentric_coordinates[2]

    @property
    def geodetic_coordinates(self):
        return self._geodetic_coordinates

    @property
    def lat(self):
        return self._geodetic_coordinates[0]

    @property
    def lon(self):
        return self._geodetic_coordinates[1]

    @property
    def height(self):
        return self._geodetic_coordinates[2]

    @property
    def earth_coordinates(self):
        return self._earth_coordinates

    @property
    def x_earth(self):
        return self._earth_coordinates[0]

    @property
    def y_earth(self):
        return self._earth_coordinates[1]

    @property
    def z_earth(self):
        return self._earth_coordinates[2]


class EarthPosition(Position):

    def __init__(self, x, y, height, lat=0, lon=0):
        # TODO: docstring
        earth = np.array([x, y, -height])
        # TODO: Assuming round earth use changes in x & y to calculate
        # new lat and lon. z_earth is -height:
        geodetic = np.array([lat, lon, height])  # m
        # TODO: make transformation from geodetic to geocentric:
        geocentric = np.zeros(3)  # m
        super().__init__(geodetic, geocentric, earth)

    def update(self, value):
        # Assuming round earth use changes in x & y to calculate
        # new lat and lon. z_earth is -height:
        delta_x, delta_y, _ = value - self.earth_coordinates
        delta_lat = delta_x / EARTH_MEAN_RADIUS
        delta_lon = delta_y / EARTH_MEAN_RADIUS
        self._geodetic_coordinates = \
            np.array([self.lat + delta_lat, self.lon + delta_lon, -value[2]])

        # TODO: make transformation from geodetic to geocentric:
        self._geocentric_coordinates = np.zeros(3)  # m

        # Update Earth coordinates with value
        self._earth_coordinates[:] = value

    def __repr__(self):
        rv = (f"x_e: {self.x_earth:.2f} m, y_e: {self.y_earth:.2f} m, "
              f"z_e: {self.z_earth:.2f} m")
        return rv


class GeodeticPosition(Position):

    def __init__(self, lat, lon, height, x_earth=0, y_earth=0):
        # TODO: docstring
        earth = np.array([x_earth, y_earth, -height])
        # TODO: Assuming round earth use changes in x & y to calculate
        # new lat and lon. z_earth is -height:
        geodetic = np.array([lat, lon, height])  # m
        # TODO: make transformation from geodetic to geocentric:
        geocentric = np.zeros(3)  # m
        super().__init__(geodetic, geocentric, earth)

    def update(self, value):
        # Assuming round earth use changes in x & y to calculate
        # new x, y from lat and lon. z_earth is -height
        delta_lat, delta_lon, _ = self.geodetic_coordinates - value
        dx_e = EARTH_MEAN_RADIUS * delta_lat
        dy_e = EARTH_MEAN_RADIUS * delta_lon
        self._earth_coordinates[:] = \
            np.array([self.x_earth + dx_e, self.y_earth + dy_e, -value[2]])

        # TODO: make transformation from geodetic to geocentric:
        self._geocentric_coordinates = np.zeros(3)  # m

        # Update geodetic coordinates with value
        self._geodetic_coordinates[:] = value


# class GeocentricPosition(Position):
    # TODO:
    # raise NotImplementedError