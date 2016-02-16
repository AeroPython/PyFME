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

    step_input = np.zeros_like(time) + A + offset
    step_input[np.where(time < t_init)] = 0
    step_input[np.where(time > t_init + T)] = 0
    step_input = step_input + var

    return (step_input)
