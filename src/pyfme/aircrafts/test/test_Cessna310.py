# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 23:56:51 2016

@author:olrosales@gmail.com

@AeroPython
"""

import pytest

import numpy as np

from numpy.testing import (assert_array_almost_equal, assert_almost_equal)

from pyfme.aircrafts.Cessna310 import (Mass_and_Inertial_Data,
                                       q, get_forces, get_moments)




def test_q():
    
    U = 100
    rho = 1.225
    q_expected = 6125
    
    dinamyc_pressure = q(U, rho)
    
    assert_almost_equal(dinamyc_pressure, q_expected)
    
def test_get_forces():
    
    U = 100
    rho = 1.225
    alpha = 0
    beta = 0
    deltae = 0
    ih = 0
    deltaail = 0
    deltar = 0
    forces_expected = np.array([-2887.82725, -28679.112, 0])
        
    forces = get_forces( U, rho, alpha, beta, deltae, ih, deltaail, deltar)
    
    assert_array_almost_equal(forces, forces_expected)
    
def test_get_moments():
    
    U = 100
    rho = 1.225
    alpha = 0
    beta = 0
    deltae = 0
    ih = 0
    deltaail = 0
    deltar = 0
    moments_expected = np.array([0, 3101.963823, 0])
        
    moments = get_moments( U, rho, alpha, beta, deltae, ih, deltaail, deltar)
    
    assert_array_almost_equal(moments, moments_expected)
    
   