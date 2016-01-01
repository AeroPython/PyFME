# -*- coding: utf-8 -*-
"""
Flight Dynamic Equations of Motion

These are the equations to be integrated, thus they have the following order
for the arguments:
func(time, y, ...) where dy/dt = func(y, ...)

Assumptions:

* ...
"""

import numpy as np


def linear_and_angular_momentum_eqs(time, vel, mass, inertia, forces, moments):
    """Linear and angular momentum equations

     Parameters
    ----------
    time : float
        Current time (s).
    vel : array_like
        Current value of absolute velocity and angular velocity, both expressed
        in body axes (u, v, w, p, q, r) in (m/s, m/s, m/s, rad/s, rad/s rad/s).
    mass : float
        Current mass of the aircraft (kg).
    inertia : array_like
        3x3 tensor of inertia of the aircraft (kg · m²)
        Current equations assume that the aircraft has a symmetry plane
        (x_b - z_b), thus J_xy and J_yz must be null.
    forces : array_like
        3 dimensional vector containing the total forces (including gravity) in
        x_b, y_b, z_b axes (N).
    moments : array_like
        3 dimensional vector containing the total moments in x_b, y_b, z_b axes
        (N·m).

    Returns
    -------


    Raises
    ------


    See Also
    --------

    Notes
    -----


    References
    ----------
    .. [1]
    """

    Ix = inertia[0, 0]
    Iy = inertia[1, 1]
    Iz = inertia[2, 2]

    # Note definition of moments of inertia p.21 Gomez Tierno, et al Mecánica
    # de vuelo
    # TODO: define moments of inertia like this for all the code.
    Jxz = - inertia[0, 2]

    Fx, Fy, Fz = forces

    L, M, N = moments

    u, v, w = vel[0:3]
    p, q, r = vel[3:6]

    # Linear momentum equations
    du_dt = Fx / mass + r * v - q * w
    dv_dt = Fy / mass - r * u + p * w
    dw_dt = Fz / mass + q * u - p * v

    # Angular momentum equations
    dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
             p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)

    dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy

    dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
             q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)


    return np.array([du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt])

def jac_linear_and_angular_momentum_eqs(time, vel, mass, inertia):
    """ Jacobian of linear and angular momentum equations
    """
    Ix = inertia[0, 0]
    Iy = inertia[1, 1]
    Iz = inertia[2, 2]

    Jxz = - inertia[0, 2]

    u, v, w = vel[0:3]
    p, q, r = vel[3:6]

    jac = np.zeros([6,6])

    jac[0, 1] = r
    jac[0, 2] = - q
    jac[0, 4] = - w
    jac[0, 5] = v

    jac[1, 0] = - r
    jac[1, 2] = p
    jac[1, 3] = w
    jac[1, 5] = - u

    jac[2, 0] = q
    jac[2, 1] = - p
    jac[2, 3] = - v
    jac[2, 4] = u

    jac[3, 3] = Jxz * q * (Ix + Iz - Iy) / (Ix * Iz - Jxz ** 2)
    jac[3, 4] = (p * Jxz * (Ix + Iz - Iy) - r * (Iz ** 2 - Iz * Iy + Jxz ** 2))\
                / (Ix * Iz - Jxz ** 2)
    jac[3, 5] = - q * (Iz ** 2 - Iz * Iy + Jxz ** 2) / (Ix * Iz - Jxz ** 2)

    jac[4, 3] = ((Iz - Ix) * r - 2 * Jxz * p) / Iy
    jac[4, 5] = ((Iz - Ix) * p + 2 * Jxz * r) / Iy

    jac[5, 3] = q * (Ix ** 2 - Ix * Iy + Jxz ** 2) / (Ix * Iz - Jxz ** 2)
    jac[5, 4] = (p * (Ix ** 2 - Ix * Iy + Jxz ** 2) - r * Jxz * (Ix + Iz - Iy))\
               / (Ix * Iz - Jxz ** 2)
    jac[5, 5] = - q * Jxz * (Iz + Ix - Iy) / (Ix * Iz - Jxz ** 2)

    return jac




def kinematic_angular_eqs(time, euler_angles, angular_vel):
    """ Kinematic angular equations
    """

    theta, phi, psi = euler_angles
    p, q, r = angular_vel

    dtheta_dt = q * np.cos(phi) - r * np.sin(phi)

    dphi_dt = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)

    dpsi_dt = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    return np.array([dtheta_dt, dphi_dt, dpsi_dt])


def jac_kinematic_angular_eqs(time, euler_angles, angular_vel):
    """Jacobian of kinematic angular equations
    """
    theta, phi, psi = euler_angles
    p, q, r = angular_vel

    jac = np.zeros([3,3])

    jac[0, 1] = - (q * np.sin(phi) + r * np.cos(phi))

    jac[1, 0] = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta) ** 2
    jac[1, 1] = (q * np.cos(phi) - r * np.sin(phi)) * np.tan(theta)

    jac[2, 0] = - (q * np.sin(phi) + r * np.cos(phi)) * (np.tan(theta) /
                   np.cos(theta))
    jac[2, 1] = (q * np.cos(phi) - r * np.sin(phi)) / np.cos(theta)

    return jac



def navigation_eqs(time, pos, vel, euler_angles):
    """Kinematic linear equations
    """

    u, v, w = vel
    theta, phi, psi = euler_angles

    dx_dt = np.cos(theta) * np.cos(phi) * u + (np.sin(phi) * np.sin(theta) *
            np.cos(psi) - np.cos(phi) * np.sin(psi)) * v + (np.cos(phi) *
            np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w

    dy_dt = np.cos(theta) * np.sin(psi) * u + (np.sin(phi) * np.sin(theta) *
            np.sin(psi) +  np.cos(phi) * np.cos(psi)) * v + (np.cos(phi) *
            np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w

    dz_dt = - np.sin(theta) * u + np.sin(phi) * np.cos(theta) * v + \
            np.cos(phi) * np.cos(theta) * w

    return np.array([dx_dt, dy_dt, dz_dt])
