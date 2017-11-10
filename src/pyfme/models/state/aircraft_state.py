# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Aircraft State
--------------

"""

from .angular_velocity import BodyAngularVelocity
from .acceleration import BodyAcceleration
from .angular_acceleration import BodyAngularAcceleration


class AircraftState:
    def __init__(self, position, attitude, velocity, angular_vel=None,
                 acceleration=None, angular_accel=None):

        self.position = position
        self.attitude = attitude
        self.velocity = velocity

        if angular_vel is None:
            angular_vel = BodyAngularVelocity(0, 0, 0, attitude)
        if acceleration is None:
            acceleration = BodyAcceleration(0, 0, 0, attitude)
        if angular_accel is None:
            angular_accel = BodyAngularAcceleration(0, 0, 0, attitude)

        self.angular_vel = angular_vel
        self.acceleration = acceleration
        self.angular_accel = angular_accel

    def __repr__(self):
        rv = (
            "Aircraft State \n"
            f"{self.position} \n"
            f"{self.attitude} \n"
            f"{self.velocity} \n"
            f"{self.angular_vel} \n"
            f"{self.acceleration} \n"
            f"{self.angular_accel} \n"
        )
        return rv
