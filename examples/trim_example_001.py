# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

<Module name>
<Brief description ...>
"""
from pyfme.aircrafts import  cessna_310
from pyfme.utils.trimmer import steady_state_flight_trim
from pyfme.utils.coordinates import wind2body

TAS = 120  # m/s
h = 1000

results = steady_state_flight_trim(cessna_310, h, TAS, gamma=0, turn_rate=0)
# lin_vel, ang_vel, theta, phi, alpha, beta, control_vector

alpha = results[3]
beta = results[4]
lin_vel = wind2body((TAS, 0, 0), alpha, beta)

print('')
print('Results:')
print("lin_vel = {}".format(lin_vel))
print("ang_vel = {}".format(results[0]))
print("theta = {}".format(results[1]))
print("phi = {}".format(results[2]))
print("alpha = {}".format(results[3]))
print("beta = {}".format(results[4]))
print("control = {}".format(results[5]))
print('')

