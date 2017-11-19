"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""


class Environment(object):
    """
    Stores all the environment info: atmosphere, gravity and wind.
    """

    def __init__(self, atmosphere, gravity, wind):
        """
        Parameters
        ----------
        atmosphere : Atmosphere
            Atmospheric model.
        gravity : Gravity
            Gravity model.
        wind : Wind
            Wind or gust model.
        """
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
        return self.wind.body

    @property
    def horizon_wind(self):
        return self.wind.horizon

    def update(self, state):
        self.atmosphere.update(state)
        self.gravity.update(state)
        self.wind.update(state)
