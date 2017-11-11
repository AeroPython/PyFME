# -*- coding: utf-8 -*-
"""
Tests of equations of euler flat earth model.
"""

import pytest
import numpy as np

from pyfme.models.euler_flat_earth import _system_equations, EulerFlatEarth
from pyfme.models.state import (EarthPosition, EulerAttitude, BodyVelocity,
                                BodyAngularVelocity, BodyAcceleration,
                                BodyAngularAcceleration, AircraftState)
from pyfme.models.constants import EARTH_MEAN_RADIUS


pos = EarthPosition(0, 0, 2000)
att = EulerAttitude(5/180*np.pi, 15/180*np.pi, 45/180*np.pi)
vel = BodyVelocity(50, 2, 3, att)
ang_vel = BodyAngularVelocity(1/180*np.pi, 5/180*np.pi, 5/180*np.pi, att)
accel = BodyAcceleration(0, 0, 0, att)
ang_accel = BodyAngularAcceleration(0, 0, 0, att)

full_state = AircraftState(pos, att, vel, ang_vel, accel, ang_accel)


def test_system_equations():
    time = 0
    state_vector = np.array(
        [1, 1, 1, 1, 1, 1,
         np.pi / 4, np.pi / 4, 0,
         1, 1, 1],
        dtype=float
    )

    mass = 10
    inertia = np.array([[1000,    0, -100],
                        [   0,  100,    0],
                        [-100,    0,  100]], dtype=float)

    forces = np.array([100., 100., 100.], dtype=float)
    moments = np.array([100., 1000., 100], dtype=float)

    exp_sol = np.array(
        [10, 10, 10, 11. / 9, 1, 92. / 9,
         0, 1 + 2 ** 0.5, 2,
         1 + (2 ** 0.5) / 2, 0, 1 - (2 ** 0.5) / 2],
        dtype=float
    )
    sol = _system_equations(time, state_vector, mass, inertia, forces, moments)
    np.testing.assert_allclose(sol, exp_sol, rtol=1e-7, atol=1e-15)


def test_fun_raises_error_if_no_update_simulation_is_defined():
    system = EulerFlatEarth(t0=0, full_state=full_state)
    x = np.zeros_like(system.state_vector)
    with pytest.raises(TypeError):
        system.fun(t=0, x=x)


def test_update_full_system_state_from_state():
    system = EulerFlatEarth(t0=0, full_state=full_state)

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    x_dot = np.array([13, 14, 15, 16, 17, 18])


    # Lat and lon after update in EarthFlatEarth are calculated from
    # delta_x, and delta_y with Earth mean radius, so depends on the
    # previous state.
    # If this test broke after changing the way lat and lon are updated from
    # Earth coordinates (ie. taking into account only the first point and
    # current status) you are in the right place
    dlat = (x[9] - system.full_state.position.x_earth) / EARTH_MEAN_RADIUS
    dlon = (x[10] - system.full_state.position.y_earth) / EARTH_MEAN_RADIUS

    system._update_full_system_state_from_state(x, x_dot)

    exp_pos = EarthPosition(10, 11, -12, lat=dlat, lon=dlon)
    exp_att = EulerAttitude(7, 8, 9)
    exp_vel = BodyVelocity(1, 2, 3, exp_att)
    exp_ang_vel = BodyAngularVelocity(4, 5, 6, exp_att)
    exp_accel = BodyAcceleration(13, 14, 15, exp_att)
    exp_ang_accel = BodyAngularAcceleration(16, 17, 18, exp_att)

    exp_full_state = AircraftState(exp_pos, exp_att, exp_vel, exp_ang_vel,
                                   exp_accel, exp_ang_accel)

    for ii, jj in zip(system.full_state._value, exp_full_state._value):
        print(ii, jj)

    np.testing.assert_allclose(system.full_state._value, exp_full_state._value)


def test_get_state_vector_from_full_state():

    system = EulerFlatEarth(0, full_state)

    x = np.array([50, 2, 3,
                  1/180*np.pi, 5/180*np.pi, 5/180*np.pi,
                  5/180*np.pi, 15/180*np.pi, 45/180*np.pi,
                  0, 0, -2000])

    np.testing.assert_allclose(system.state_vector, x)
