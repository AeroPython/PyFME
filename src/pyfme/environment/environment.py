"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""
from pyfme.environment.atmosphere import Atmosphere
from pyfme.models.systems import GeneralSystem


class Environment(object):

    def __init__(self, atmosphere: Atmosphere, gravity, wind):
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
        return self.a
    
    
    def update(self, system: GeneralSystem):
        self.atmosphere.update(system)
        self.gravity.update(system)
        self.wind.update(system)
        pass