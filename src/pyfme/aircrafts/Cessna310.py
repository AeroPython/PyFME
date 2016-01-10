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
    
    Returns
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
    
    Returns
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
    
    """Assigns the value of the coefficients
    of stability in cruise conditions and order them in a matrix.
    
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
    
    
def Lat_Aero_Coefficients():
    
    """Assigns the value of the coefficients
    of stability in cruise conditions and order them in a matrix.
    
    Coefficients
    ------------
    
    CYb is the side force stability derivative with respect to the  
        angle of sideslip
    CYda is the side force stability derivative with respect to the 
         aileron deflection
    CYdr is the side force stability derivative with respect to the 
         rudder deflection
    
    Clb is the rolling moment stability derivative with respect to 
        angle of sideslip
    Clda is the rolling moment stability derivative with respect to 
        the aileron deflection
    Cldr is the rolling moment stability derivative with respect to 
        the rudder deflection
    
    Cnb is the yawing moment stability derivative with respect to the 
        angle of sideslip
    Cnda is the yawing moment stability derivative with respect to the 
        aileron deflection
    Cndr is the yawing moment stability derivative with respect to the 
        rudder deflection
   
        
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 590
    """
   
    CYb = -0.698
    CYda = 0
    CYdr = 0.230
    
    Clb = -0.1096
    Clda = 0.172
    Cldr = 0.0192
    
    Cnb = 0.1444
    Cnda = -0.0168
    Cndr = -0.1152
    
    Lat_coef_matrix = np.array([
                                [CYb, CYda, CYdr],
                                [Clb, Clda, Cldr],
                                [Cnb, Cnda, Cndr]
                                ])
    
    return Lat_coef_matrix
    
def q (U,rho):
    
    """ Calculates  the dinamic pressure q = 0.5*rho*U^2
    
    Parameters
    ----------
    
    rho : float
          density (SI)
    U : flota
        velocity (SI)
        
    Returns
    -------
    
    q : float
        dinamic pressure 
        
    """
    #Aquí falta asegurarse que la densidad y la velocidad que entra son correctas    
    
    q = 0.5 * rho * (U ** 2)
    
    return q
    
    
    
def get_forces( U, rho, alpha, beta, deltae, ih, deltaail, deltar):
    
    
    """ Calculates forces 
    
    Parameters
    ----------
    
    rho = float
          density (SI)
    U = flota
        velocity (SI)
    
    alpha : float
            attack angle (rad).
    beta : float
           sideslip angle (rad).
    deltae : float
             elevator deflection (rad).
    ih : float
         stabilator deflection (rad).
    deltaail : float
               aileron deflection (rad).
    deltar : float
             rudder deflection (rad).
    
    
    
        
    Returns
    -------
    
    forces : array_like
             3 dimensional vector with (Fxs, Fys, Fzs) forces in stability axes.
        
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    chapter 3 and 4 
    """
 
    Long_coef_matrix = Long_Aero_Coefficients()
    Lat_coef_matrix = Lat_Aero_Coefficients()
    
    
    CL0, CLa, CLde, CLdih = Long_coef_matrix[0,:]
    CD0, CDa, CDde, CDdih = Long_coef_matrix[1,:]
    CYb, CYda, CYdr = Lat_coef_matrix[0,:]
    

    
    
    CLfull = CL0 + CLa * alpha + CLde * deltae + CLdih * ih
    CDfull = CD0 + CDa * alpha + CDde * deltae + CDdih * ih  
    CYfull = CYb * beta + CYda* deltaail + CYdr * deltar
 

    forces = q(U,rho) * Geometric_Data()[0] * np.array([-CDfull, -CLfull, CYfull])
    
    return forces
             
             
def get_moments( U, rho, alpha, beta, deltae, ih, deltaail, deltar):
    
    
    """ Calculates forces 
    
    Parameters
    ----------
    
    rho = float
          density (SI)
    U = flota
        velocity (SI)
    
    alpha : float
            attack angle (rad).
    beta : float
           sideslip angle (rad).
    deltae : float
             elevator deflection (rad).
    ih : float
         stabilator deflection (rad).
    deltaail : float
               aileron deflection (rad).
    deltar : float
             rudder deflection (rad).
    
    
    
        
    returns
    -------
    
    moments : array_like
             3 dimensional vector with (Mxs, Mys, Mzs) forces in stability axes.
        
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    chapter 3 and 4 
    """

    Long_coef_matrix = Long_Aero_Coefficients()
    Lat_coef_matrix = Lat_Aero_Coefficients()

    
    Cm0, Cma, Cmde, Cmdih = Long_coef_matrix[2,:]
    Clb, Clda, Cldr = Lat_coef_matrix[1,:]
    Cnb, Cnda, Cndr = Lat_coef_matrix[2,:]
    
    
    Cmfull = Cm0 + Cma * alpha + Cmde * deltae + Cmdih * ih
    Clfull = Clb * beta + Clda* deltaail + Cldr * deltar
    Cnfull = Cnb * beta + Cnda* deltaail + Cndr * deltar 

    moments = q(U,rho) * Geometric_Data()[0] * np.array([Clfull * Geometric_Data()[2],
                                                      Cmfull * Geometric_Data()[1],
                                                      Cnfull * Geometric_Data()[2]
                                                      ])
    
    return moments            
    
    
    
    
a = get_moments(100, 1.225, 0, 0 , 0 ,0, 0 ,0)        
    
    
    
    
    
    
    
    
    
    


    

