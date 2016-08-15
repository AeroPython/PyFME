"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Wind Models
-----------

"""
import numpy as np

class NoWind(object):

    def __init__(self):
        # Wind velocity: FROM North to South, FROM East to West,
        # Wind velocity in the UPSIDE direction
        self.horizon_wind = np.zeros([3], dtype=float)
        self.body_wind = np.zeros([3], dtype=float)

    def update(self, system):
        pass
