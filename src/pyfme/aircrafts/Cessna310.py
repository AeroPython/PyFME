# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:44:39 2016

@author:olrosales@gmail.com 

@AeroPython
"""

## Aircraft = Cessna 310


import numpy as np

def Geometric_Data():
    
    """ Provides the value of some geometric data.
    
    Data
    ----
    
    Sw    Surface (m^2)
    c  Mean Aerodynamic Chord (m2)
    b    Wing Span (m2)
    
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """
    #falta expresar el valor de conversión entre el libro y el SI
    Sw = 16.258
    c = 0.4450056
    b = 3.428122
    
    return Sw, c, b

def Mass_and_Inertial_Data():
    
    """ Provides the value of some mass and inertial data.
    
    Data
    -----
    m   mass (lb * 0.453592 = kg)
    Ixxb Moment of Inertia x-axis ( slug * ft2 * 1.3558179 = Kg * m2)
    Iyyb Moment of Inertia y-axis ( slug * ft2 * 1.3558179 = Kg * m2)
    Izzb Moment of Inertia z-axis ( slug * ft2 * 1.3558179 = Kg * m2)
    Ixzb Product of Inertia xz-plane ( slug * ft2 * 1.3558179 = Kg * m2)
    
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """

    m = 4600 * 0.453592
    Ixxb = 8884 * 1.3558179
    Iyyb = 1939 * 1.3558179
    Izzb = 11001 * 1.3558179
    Ixzb = 0 * 1.3558179
    
    I_matrix = np.diag([Ixxb,Iyyb,Izzb])
    
    return m, I_matrix

def Long_Aero_Coefficients():
    
    """Long_Aero_coefficients assigns the value of the coefficients
    of stability and order them in a matrix.
    
    Coefficients
    ------------
    
    CL0 is the lift coefficient evaluated at the initial condition
    CLa is the lift stability derivative with respect to the angle of attack
    CLde is the lift stability derivative with respect to the elevator 
         deflection
    CLdih is the lift stability derivative with respect to the stabilator 
         deflection
    
    CD0 is the drag coefficiente evaluated at the initial condition
    CDa is the drag stability derivative with respect to the angle of attack
    
    Cm0 is the pitching moment coefficient evaluated at the condition 
        (alpha0 = deltaE = deltaih = 0º)
    Cma is the pitching moment stability derivative with respect to the angle
        of attack
    Cmde is the pitching moment stability derivative with respect to the 
        elevator deflection
    Cmdih is the pitching moment stability derivative with respect to the 
         stabilator deflection
        
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """
    
    
    
    
    CL0 = 0.288
    CLa = 4.58
    CLde = 0.81
    CLdih = 0
    
    CD0 = 0.029
    CDa = 0.160
    
    Cm0 = 0.07
    Cma = -0.137
    Cmde = -2.26
    Cmdih = 0 
    
    Long_coef_matrix = np.array([
    [CL0, CLa, CLde, CLdih],
    [CD0, CDa, 0, 0],
    [Cm0, Cma, Cmde, Cmdih]
    ])
    
    return Long_coef_matrix
    
    
    




