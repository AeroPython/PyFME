"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

OOP Example
-----------

"""

from pyfme.aircrafts import Cessna310

from pyfme import Environment
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.wind import NoWind

from pyfme.models import EulerFlatEarth

aircraft = Cessna310()
environment = Environment(ISA1976, VerticalConstant, NoWind)
system = EulerFlatEarth()

