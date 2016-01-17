# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:32:11 2016

@author: Juan
"""
import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal)


from pyfme.utils.coordinates import (quatern2euler, euler2quatern,
                                     check_unitnorm)


def test_quatern2euler():

    quaternion = np.array([0.8660254037844387, 0, 0.5, 0])

    euler_angles_expected = np.array([0.0, 1.0471975511965976, 0.0])

    euler_angles = quatern2euler(quaternion)

    assert_array_almost_equal(euler_angles, euler_angles_expected)


print(quatern2euler(0.8660254037844387, 0, 0.5, 0))

print(euler2quatern(0.0, 1.0471975511965976, 0.0))
