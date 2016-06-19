"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Generic Aircraft
----------------

"""
from abc import abstractmethod
import numpy as np
from scipy.optimize import least_squares

from pyfme.environment.environment import Environment
from pyfme.models.euler_flat_earth import lamceq
from pyfme.models.systems import System


class Aircraft(object):

    def __init__(self):
        pass

    @property
    def Ixx(self):
        return self.inertia[0, 0]

    @property
    def Iyy(self):
        return self.inertia[1, 1]

    @property
    def Izz(self):
        return self.inertia[2, 2]

    @abstractmethod
    def get_forces_and_moments(self, system: System, controls: dict,
                               env: Environment):
        pass

    @abstractmethod
    def check_control_limits(self):
        pass

    def steady_state_flight_trim(self, system: System, env: Environment,
                             controls: dict, trim_controls_names: list,
                             verbose=0):
        # TODO: wrap steady_state_flight_trim from utils/trimmer.py
        raise NotImplementedError