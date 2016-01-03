# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 23:07:51 2016

@author: asaez
"""

import numpy as np
import matplotlib.pyplot as plt

from models.system import System


def forces():
    return np.array([0, 0, -9.8*500])


def moments():
    return np.array([0, 0, 0])

# Case params
mass = 500
inertia = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

# initial conditions
u, v, w = 0, 0, 0
p, q, r = 0, 0, 0

theta, phi, psi = 0, 0, 0
x, y, z = 0, 0, 5000

t0 = 0
tf = 10
dt = 1e-2
time = np.arange(t0, tf, dt)

results = np.zeros([time.size, 12])

velocities = np.empty([time.size, 6])
results[0, 0:6] = u, v, w, p, q, r
results[0, 6:9] = theta, phi, psi
results[0, 9:12] = x, y, z

# Linear Momentum and Angular Momentum eqs
equations = System(integrator='dopri5', model='euler_flat_earth', jac=False)
equations.set_initial_values(u, v, w, p, q, r, theta, phi, psi, x, y, z)


for ii, t in enumerate(time[1:]):
    results[ii+1, :] = equations.propagate(mass, inertia, forces(), moments(),
                                           dt=dt)

velocities = results[:, 0:6]
attitude_angles = results[:, 6:9]
position = results[:, 9:12]


plt.figure('pos')
plt.plot(time, position[:, 0])
plt.plot(time, position[:, 1])
plt.plot(time, position[:, 2])

plt.figure('angles')
plt.plot(time, attitude_angles[:, 0])
plt.plot(time, attitude_angles[:, 1])
plt.plot(time, attitude_angles[:, 2])

plt.figure('velocities')
plt.plot(time, velocities[:, 0])
plt.plot(time, velocities[:, 1])
plt.plot(time, velocities[:, 2])

plt.figure('ang velocities')
plt.plot(time, velocities[:, 3])
plt.plot(time, velocities[:, 4])
plt.plot(time, velocities[:, 5])

