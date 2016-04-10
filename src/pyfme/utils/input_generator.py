# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Inputs generator
-------------
Provides some typical inputs
"""
import numpy as np


def step(t_init, T, A, time, offset=0, var=None):
    """ Step input

    Parameters
    ----------
    t_init : float
        time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude
    time : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    step_input : array_like
    """
    if var is None:
        step_input = np.ones_like(time) * float(offset)
        step_input[(time >= t_init) & (time <= t_init + T)] += A

    else:
        if np.size(var) == np.size(time):
            step_input = float(offset) + var
            step_input[(time >= t_init) & (time <= t_init + T)] += A

        else:
            raise ValueError('var and time must have the same size')

    return step_input


def doublet(t_init, T, A, time, offset=0, var=None):

    """ doublet input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude
    time : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    doublet_input : array_like
    """
    if var is None:

        doublet_input = np.ones_like(time) * float(offset)
        doublet_input[(time >= t_init) & (time <= t_init + T/2)] += A/2
        doublet_input[(time > t_init + T/2) & (time <= t_init + T)] += - A/2

    else:

        if np.size(var) == np.size(time):
            doublet_input = float(offset) + var
            doublet_input[(time >= t_init) & (time <= t_init + T/2)] += A/2
            doublet_input[(time > t_init + T/2) & (time <= t_init + T)] += - A/2

        else:

            raise ValueError('var and time must have the same size')

    return (doublet_input)


def sinusoide(t_init, T, A, time, phase=0, offset=0, var=None):

    """ sinusoide input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad)
    time : array_like
        time simulation vector (s)
    phase : float
        sinusoidal phase (rad)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    sinusoide_input : array_like
    """

    time_input = time[(time >= t_init) & (time <= t_init + T)]

    if var is None:

        sinusoide_input = np.ones_like(time) * float(offset)
        sinusoide_input[(time >= t_init) & (time <= t_init + T)] += A/2 *\
        np.sin(2 * np.pi * (time_input - t_init) / T + phase)

    else:

        if np.size(var) == np.size(time):

            sinusoide_input = var + float(offset)
            sinusoide_input[(time >= t_init) & (time <= t_init + T)] += A/2 *\
            np.sin(2 * np.pi * (time_input - t_init) / T + phase)

        else:

            raise ValueError('var and time must have the same size')

    return (sinusoide_input)


def ramp(t_init, T, A, time, offset=0, var=None):

    """ ramp input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad)
    time : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    ramp_input : array_like
    """

    time_input = time[(time >= t_init) & (time <= t_init + T)]

    if var is None:

        ramp_input = np.ones_like(time) * float(offset)
        ramp_input[(time >= t_init) & (time <= t_init + T)] += (A / T) *\
        (time_input - t_init)

    else:

        if np.size(var) == np.size(time):

            ramp_input = var + offset
            ramp_input[(time >= t_init) & (time <= t_init + T)] += (A / T) *\
            (time_input - t_init)

        else:

            raise ValueError('var and time must have the same size')

    return (ramp_input)


def harmonic(t_init, T, A, time, f, phase=0, offset=0, var=None):

    """ sinusoide input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad)
    time : array_like
        time simulation vector (s)
    f : float
        sinusoidal frequency (s)
    phase : float
        sinusoidal phase (rad)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    harmonic_input : array_like
    """
    time_input = time[(time >= t_init) & (time <= t_init + T)]

    if var is None:

        harmonic_input = np.ones_like(time) * float(offset)
        harmonic_input[(time >= t_init) & (time <= t_init + T)] += A/2 *\
        np.sin(2 * np.pi * f * (time_input - t_init) + phase)

    else:

        if np.size(var) == np.size(time):

            harmonic_input = var + float(offset)
            harmonic_input[(time >= t_init) & (time <= t_init + T)] += A/2 *\
            np.sin(2 * np.pi * f * (time_input - t_init) + phase)
        else:

            raise ValueError('var and time must have the same size')

    return (harmonic_input)
