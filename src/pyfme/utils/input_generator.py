# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Inputs generator
----------------
Provides some typical inputs signals such as: step, doublet, ramp, harmonic.
"""
from abc import abstractmethod

from numpy import sin, pi


# TODO: documentation
class Control(object):

    def __init__(self):
        self._fun = None

    @abstractmethod
    def __call__(self, t):
        raise NotImplementedError

    def __add__(self, other):
        control = Control()
        control._fun = lambda t: self(t) + other(t)
        return control

    def __sub__(self, other):
        control = Control()
        control._fun = lambda t: self(t) - other(t)
        return control

    def __mul__(self, other):
        control = Control()
        control._fun = lambda t: self(t) * other(t)
        return control

    def __truediv__(self, other):
        control = Control()
        control._fun = lambda t: self(t) / other(t)
        return control


class Constant(Control):

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def __call__(self, t):
        return self.offset


class Step(Control):

    def __init__(self, t_init, T, A, offset=0):
        super().__init__()
        self.t_init = t_init
        self.T = T
        self.A = A
        self.offset = offset

        self.t_fin = self.t_init + self.T

    def __call__(self, t):
        value = self.offset
        if self.t_init <= t <= self.t_fin:
            value += self.A
        return value


class Doublet(Control):

    def __init__(self, t_init, T, A, offset=0):
        super().__init__()
        self.t_init = t_init
        self.T = T
        self.A = A
        self.offset = offset

        self.t_fin1 = self.t_init + self.T / 2
        self.t_fin2 = self.t_init + self.T

    def __call__(self, t):
        value = self.offset

        if self.t_init <= t < self.t_fin1:
            value += self.A / 2
        elif self.t_fin1 < t <= self.t_fin2:
            value -= self.A / 2
        return value


class Ramp(Control):

    def __init__(self, t_init, T, A, offset=0):
        super().__init__()
        self.t_init = t_init
        self.T = T
        self.A = A
        self.offset = offset

        self.slope = self.A / self.T
        self.t_fin = self.t_init + self.T

    def __call__(self, t):
        value = self.offset
        if self.t_init <= t <= self.t_fin:
            value += self.slope * (t - self.t_init)

        return value


class Harmonic(Control):

    def __init__(self, t_init, t_fin, A, freq, phase, offset=0):
        super().__init__()
        self.t_init = t_init
        self.t_fin = t_fin
        self.A = A
        self.freq = freq
        self.phase = phase
        self.offset = offset

    def __call__(self, t):
        value = self.offset

        if self.t_init <= t <= self.t_fin:
            value += self.A/2 * sin(2 * pi * self.freq * (t - self.t_init) +
                                    self.phase)

        return  value
