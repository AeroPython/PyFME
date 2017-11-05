# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Aircraft State
--------------

"""


class AircraftState:

    def __init__(self, position, attitude, velocity, angular_vel,
                 acceleration, angular_accel):

        self.position = position
        self.attitude = attitude
        self.velocity = velocity
        self.angular_vel = angular_vel
        self.acceleration = acceleration
        self.angular_accel = angular_accel
