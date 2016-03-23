# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

<Module name>
<Brief description ...>
"""
from pyfme.aircrafts import cessna_310
from pyfme.utils.trimmer import steady_state_flight_trim
from pyfme.utils.coordinates import wind2body

TAS = 80  # m/s
h = 1000

results = steady_state_flight_trim(cessna_310, h, TAS, gamma=0, turn_rate=0)
# lin_vel, ang_vel, theta, phi, alpha, beta, control_vector

alpha = results[3]
beta = results[4]
lin_vel = wind2body((TAS, 0, 0), alpha, beta)

print("""
Results
-------
Linear velocity: {lin_vel} (m/s)
Angular velocity: {ang_vel} (rad/s)
Theta, Phi: {angles} (rad)
Alpha, Beta: {wind_angles} (rad)
Control: {control}
""".format(
        lin_vel=lin_vel,
        ang_vel=results[1],
        angles=results[2:4],
        wind_angles=results[4:6],
        control=results[6]
            )
    )
