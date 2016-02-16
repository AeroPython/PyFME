# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

<Inputs generator>
-------------
<Provides some typical inputs>
< ... >
"""
import numpy as np


def step(t_init, T, A, time, offset=0, var=0):

    """ step input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad) ### Should I choose other units?
    t : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    step_input : array_like
    """

    step_input = np.zeros_like(time)
    step_input[(time >= t_init) & (time <= t_init + T)] = A + offset
    step_input = step_input + var

    return (step_input)


def doublet(t_init, T, A, time, offset=0, var=0):

    """ doublet input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad) ### Should I choose other units?
    t : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    doublet_input : array_like
    """

    doublet_input = np.zeros_like(time)
    doublet_input[(time >= t_init) & (time < t_init + T/2)] = A/2 + offset
    doublet_input[(time > t_init + T/2) & (time <= t_init + T)] = - A/2 +\
        offset
    doublet_input = doublet_input + var

    return (doublet_input)


def impulse(t_init, A, time, offset=0, var=0):

    """ impulse input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    A : float
        peak to peak amplitude (rad) ### Should I choose other units?
    t : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    impulse_input : array_like
    """

    impulse_input = np.zeros_like(time)
    impulse_input[(time == t_init)] = A + offset
    impulse_input = impulse_input + var

    return (impulse_input)


def harmonic(t_init, T, A, time, offset=0, var=0):

    """ harmonic input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad) ### Should I choose other units?
    t : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    harmonic_input : array_like
    """

    harmonic_input = np.zeros_like(time)
    harmonic_input[(time >= t_init) & (time <= t_init + T)] = A/2 * np.sin(2 *
    np.pi * (time[(time >= t_init) & (time <= t_init + T)] - t_init) / T) +\
        offset
    harmonic_input = harmonic_input + var

    return (harmonic_input)


def ramp(t_init, T, A, time, offset=0, var=0):

    """ ramp input

    Parameters
    ----------

    t_init : float
             time initial to start (s)
    T : float
        time while input is running (s)
    A : float
        peak to peak amplitude (rad) ### Should I choose other units?
    t : array_like
        time simulation vector (s)
    offset : float
    var : array_like
          vector which contains previous perturbations

    Returns
    -------
    ramp_input : array_like
    """

    ramp_input = np.zeros_like(time)
    ramp_input[(time >= t_init) & (time <= t_init + T)] = (A / T) * (time[
    (time >= t_init) & (time <= t_init + T)] - t_init) + offset
    ramp_input = ramp_input + var

    return (ramp_input)
