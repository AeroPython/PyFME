# -*- coding: utf-8 -*-
"""
The long-term objective is to a generic class System which can handle all the
models (ie. euler_flat_earth, quaternions_rotating_earth,
quaternions_rorating_earth_variable_mass...)

Meanwhile, this class can perform the propagation for the euler_flat_earth eqs
"""

import numpy as np
from scipy.integrate import ode


from models import euler_flat_earth


models = ('euler_flat_earth',)


class System(object):
    """
    Class managing the integration for the set of equations defining the model.
    """

    def __init__(self, integrator='dopri5', model='euler_flat_earth',
                 jac=False, **integrator_params):
        """
        Initialize the equations of the chosen model and selects the
        integrator. Check `scipy.integrate.ode` to see available integrators.

        If jac = True the jacobian of the equations is used for the
        integration.
        """

        if model not in models:
            raise ValueError('The specified model is not available, please \
                             check available models in ...')

        if jac:
            jac_LM_and_AM = euler_flat_earth.jac_linear_and_angular_momentum_eqs
            jac_att = euler_flat_earth.jac_linear_and_angular_momentum_eqs
            jac_nav = None  # not implemented
        else:
            jac_LM_and_AM = None
            jac_att = None
            jac_nav = None

        self._LM_and_AM_eqs = ode(euler_flat_earth.linear_and_angular_momentum_eqs,
                                  jac=jac_LM_and_AM)
        self._attitude_eqs = ode(euler_flat_earth.kinematic_angular_eqs,
                                 jac=jac_att)
        self._navigation_eqs = ode(euler_flat_earth.navigation_eqs,
                                   jac=jac_nav)

        self._LM_and_AM_eqs.set_integrator(integrator, **integrator_params)
        self._attitude_eqs.set_integrator(integrator, **integrator_params)
        self._navigation_eqs.set_integrator(integrator, **integrator_params)

        # State vector must be initialized with set_initial_values() method
        self.state_vector = None

    def set_initial_values(self, u, v, w, p, q, r, theta, phi, psi,
                           x, y, z, t0=0.0):
        """
        Set the initial values of the required variables
        """

        self._LM_and_AM_eqs.set_initial_value(y=(u, v, w, p, q, r), t=t0)
        self._attitude_eqs.set_initial_value(y=(theta, phi, psi), t=t0)
        self._navigation_eqs.set_initial_value(y=(x, y, z), t=t0)

        self.state_vector = np.array([u, v, w, p, q, r, theta, phi, psi, x, y, z])

    def propagate(self, mass, inertia, forces, moments, dt=0.1):
        """
        Performs integration step for actual_time + dt and returns the state
        vector
        """
        actual_time = self._LM_and_AM_eqs.t
        t = actual_time + dt

        self._LM_and_AM_eqs.set_f_params(mass, inertia, forces, moments)
        velocities = self._LM_and_AM_eqs.integrate(t)

        if self._LM_and_AM_eqs.successful():
            self._attitude_eqs.set_f_params(velocities[3:])
            attitude_angles = self._attitude_eqs.integrate(t)
        else:
            raise RuntimeError('Integration of Linear and angular momentum \
                                equations was not succesfull')

        if self._attitude_eqs.successful():
            self._navigation_eqs.set_f_params(velocities[0:3], attitude_angles)
            position = self._navigation_eqs.integrate(t)
        else:
            raise RuntimeError('Integration of attitude equations was not \
                                succesfull')

        if self._navigation_eqs.successful():
            self.state_vector = np.concatenate((velocities,
                                                attitude_angles,
                                                position))
        else:
            raise RuntimeError('Integration of navigation equations was not \
                                succesfull')

        return self.state_vector
