# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Inputs generator
----------------
Provides some typical inputs signals such as: step, doublet, ramp, harmonic.
"""
import numpy as np


def step(t_init, T, A, time, offset=0, var=None):
    """ Step input.

    Parameters
    ----------
    t_init : float
        Initial time (s).
    T : float
        Input signal length (s).
    A : float
        Peak to peak amplitude.
    time : array_like
        Time simulation vector (s).
    offset : float
        Signal offset.
    var : array_like, opt
        Array containing previous perturbations. Signal will be added to
        this one.

    Returns
    -------
    step_input : array_like
    """

    if var is None:
        step_input = np.zeros_like(time)
    else:
        if np.size(var) == np.size(time):
            step_input = var
        else:
            raise ValueError('var and time must have the same size')

    step_input[(time >= t_init) & (time <= t_init + T)] += A + float(offset)

    return step_input


def doublet(t_init, T, A, time, offset=0, var=None):
    """ Doublet input.

    Parameters
    ----------
    t_init : float
        Initial time (s).
    T : float
        Input signal length (s).
    A : float
        Peak to peak amplitude.
    time : array_like
        Time simulation vector (s).
    offset : float
        Signal offset.
    var : array_like, opt
        Array containing previous perturbations. Signal will be added to
        this one.

    Returns
    -------
    doublet_input : array_like
    """

    if var is None:
        doublet_input = np.zeros_like(time)
    else:
        if np.size(var) == np.size(time):
            doublet_input = var
        else:
            raise ValueError('var and time must have the same size')

    part_1 = (time >= t_init) & (time <= t_init + T / 2)
    doublet_input[part_1] += A / 2 + float(offset)

    part_2 = (time > t_init + T / 2) & (time <= t_init + T)
    doublet_input[part_2] += - A / 2 + float(offset)

    return doublet_input


def ramp(t_init, T, A, time, offset=0, var=None):
    """ Ramp input

    Parameters
    ----------
    t_init : float
        Initial time (s).
    T : float
        Input signal length (s).
    A : float
        Peak to peak amplitude.
    time : array_like
        Time simulation vector (s).
    offset : float
        Signal offset.
    var : array_like, opt
        Array containing previous perturbations. Signal will be added to
        this one.

    Returns
    -------
    ramp_input : array_like
    """

    if var is None:
        ramp_input = np.zeros_like(time)
    else:
        if np.size(var) == np.size(time):
            ramp_input = var
        else:
            raise ValueError('var and time must have the same size')

    time_input = time[(time >= t_init) & (time <= t_init + T)]
    condition = (time >= t_init) & (time <= t_init + T)
    ramp_input[condition] += (A / T) * (time_input - t_init) + float(offset)

    return ramp_input


def harmonic(t_init, T, A, time, f, phase=0, offset=0, var=None):
    """ Sinusoid input.

    Parameters
    ----------
    t_init : float
        Initial time (s).
    T : float
        Input signal length (s).
    A : float
        Peak to peak amplitude.
    time : array_like
        Time simulation vector (s).
    f : float
        Sinusoidal frequency (s).
    phase : float
        Sinusoidal phase (rad).
    offset : float
        Signal offset.
    var : array_like, opt
        Array containing previous perturbations. Signal will be added to
        this one.

    Returns
    -------
    harmonic_input : array_like
    """
    time_input = time[(time >= t_init) & (time <= t_init + T)]

    if var is None:
        harmonic_input = np.zeros_like(time)
    else:
        if np.size(var) == np.size(time):
            harmonic_input = var
        else:
            raise ValueError('var and time must have the same size')

    harmonic_input[(time >= t_init) & (time <= t_init + T)] += \
        A / 2 * np.sin(2 * np.pi * f * (time_input - t_init) + phase) + \
        float(offset)

    return harmonic_input


def sinusoid(t_init, T, A, time, phase=0, offset=0, var=None):
    """ Sinusoid input.

    Parameters
    ----------
    t_init : float
        Initial time (s).
    T : float
        Input signal length (s).
    A : float
        Peak to peak amplitude.
    phase : float
        sinusoidal phase (rad).
    time : array_like
        Time simulation vector (s).
    offset : float
        Signal offset.
    var : array_like, opt
        Array containing previous perturbations. Signal will be added to
        this one.

    Returns
    -------
    sinusoide_input : array_like
    """
    output = harmonic(t_init, T, A, time, f=1/T, phase=phase, offset=offset,
                      var=var)

    return output
