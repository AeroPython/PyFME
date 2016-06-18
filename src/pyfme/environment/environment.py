"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""
from pyfme.environment.atmosphere import Atmosphere
from pyfme.environment.gravity import Gravity
from pyfme.environment.wind import NoWind
from pyfme.models.systems import System


class Environment(object):

    def __init__(self, atmosphere: Atmosphere, gravity: Gravity, wind: NoWind):
        self.atmosphere = atmosphere
        self.gravity = gravity
        self.wind = wind
    
    @property
    def T(self):
        return self.atmosphere.T

    @property
    def p(self):
        return self.atmosphere.p

    @property
    def rho(self):
        return self.atmosphere.rho

    @property
    def a(self):
        return self.atmosphere.a

    @property
    def gravity_magnitude(self):
        return self.gravity.magnitude

    @property
    def gravity_vector(self):
        return self.gravity.vector

    @property
    def body_wind(self):
        return self.wind.body_wind

    @property
    def horizon_wind(self):
        return self.wind.horizon_wind

    def update(self, system: System):
        self.atmosphere.update(system)
        self.gravity.update(system)
        self.wind.update(system)