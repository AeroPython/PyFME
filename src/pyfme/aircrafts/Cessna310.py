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
    
    Sw : float
         Wing surface (ft2 * 0.3048 * 0.3048 = m2)
    c : foat
        Mean aerodynamic Chord (ft * 0.3048 = m)
    b : float 
         Wing span (ft * 0.3048 = m)
    
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """
    
    Sw = 175 * 0.3048 * 0.3048    # m2
    c = 4.79 * 0.3048   # m
    b = 36.9 * 0.3048   # m
    
    return Sw, c, b

def Mass_and_Inertial_Data():
    
    """ Provides the value of some mass and inertial data.
    
    Returns
    ------
    m : float
        mass (lb * 0.453592 = kg)
    I_xx_b : float
             Moment of Inertia x-axis ( slug * ft2 * 1.3558179 = Kg * m2)
    I_yy_b : float
             Moment of Inertia y-axis ( slug * ft2 * 1.3558179 = Kg * m2)
    I_zz_b : float
             Moment of Inertia z-axis ( slug * ft2 * 1.3558179 = Kg * m2)
    I_xz_b : float
             Product of Inertia xz-plane ( slug * ft2 * 1.3558179 = Kg * m2)
    I_matrix : array_like
               I_xx_b,I_yy_b,I_zz_b]
              
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """

    m = 4600 * 0.453592   # kg
    I_xx_b = 8884 * 1.3558179   # Kg * m2
    I_yy_b = 1939 * 1.3558179   # Kg * m2
    I_zz_b = 11001 * 1.3558179   # Kg * m2
    I_xz_b = 0 * 1.3558179   # Kg * m2
    
    I_matrix = np.diag([I_xx_b,I_yy_b,I_zz_b])
    
    return m, I_matrix

def Long_Aero_Coefficients():
    
    """Assigns the value of the coefficients
    of stability in cruise conditions and order them in a matrix.

    
    C_L_0 is the lift coefficient evaluated at the initial condition
    C_L_a is the lift stability derivative with respect to the angle of attack
    C_L_de is the lift stability derivative with respect to the elevator 
         deflection
    C_L_dih is the lift stability derivative with respect to the stabilator 
         deflection
    
    C_D_0 is the drag coefficient evaluated at the initial condition
    C_D_a is the drag stability derivative with respect to the angle of attack
    
    C_m_0 is the pitching moment coefficient evaluated at the condition 
        (alpha0 = deltaE = deltaih = 0º)
    C_m_a is the pitching moment stability derivative with respect to the angle
        of attack
    C_m_de is the pitching moment stability derivative with respect to the 
        elevator deflection
    C_m_dih is the pitching moment stability derivative with respect to the 
         stabilator deflection

    Returns 
    -------    
    
    Long_coef_matrix : array_like
                                [
                                [C_L_0, C_L_a, C_L_de, C_L_dih],
                                [C_D_0, C_D_a, 0, 0],
                                [C_m_0, C_m_a, C_m_de, C_m_dih]
                                ]
                       
        
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 589
    """
    C_L_0 = 0.288
    C_L_a = 4.58
    C_L_de = 0.81
    C_L_dih = 0
    
    C_D_0 = 0.029
    C_D_a = 0.160
    
    C_m_0 = 0.07
    C_m_a = -0.137
    C_m_de = -2.26
    C_m_dih = 0 
    
    Long_coef_matrix = np.array([
                                [C_L_0, C_L_a, C_L_de, C_L_dih],
                                [C_D_0, C_D_a, 0, 0],
                                [C_m_0, C_m_a, C_m_de, C_m_dih]
                                ])
    
    return Long_coef_matrix
       
def Lat_Aero_Coefficients():
    
    """Assigns the value of the coefficients
    of stability in cruise conditions and order them in a matrix.
    
    C_Y_b is the side force stability derivative with respect to the  
        angle of sideslip
    C_Y_da is the side force stability derivative with respect to the 
         aileron deflection
    C_Y_dr is the side force stability derivative with respect to the 
         rudder deflection
    
    C_l_b is the rolling moment stability derivative with respect to 
        angle of sideslip
    C_l_da is the rolling moment stability derivative with respect to 
        the aileron deflection
    C_l_dr is the rolling moment stability derivative with respect to 
        the rudder deflection
    
    C_n_b is the yawing moment stability derivative with respect to the 
        angle of sideslip
    C_n_da is the yawing moment stability derivative with respect to the 
        aileron deflection
    C_n_dr is the yawing moment stability derivative with respect to the 
        rudder deflection
        
    returns
    -------
    Long_coef_matrix : array_like
                                [
                                [CYb, CYda, CYdr],
                                [Clb, Clda, Cldr],
                                [Cnb, Cnda, Cndr]
                                ]
    
      
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    page 590
    """
   
    C_Y_b = -0.698
    C_Y_da = 0
    C_Y_dr = 0.230
    
    C_l_b = -0.1096
    C_l_da = 0.172
    C_l_dr = 0.0192
    
    C_n_b = 0.1444
    C_n_da = -0.0168
    C_n_dr = -0.1152
    
    Lat_coef_matrix = np.array([
                                [C_Y_b, C_Y_da, C_Y_dr],
                                [C_l_b, C_l_da, C_l_dr],
                                [C_n_b, C_n_da, C_n_dr]
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
    
    
    
def get_aerodynamic_forces( U, rho, alpha, beta, delta_e, ih, delta_ail, delta_r):
    
    """ Calculates forces 
    
    Parameters
    ----------
    
    rho : float
          density (kg/(m3))
    U : float
        velocity (m/s)
    
    alpha : float
            attack angle (rad).
    beta : float
           sideslip angle (rad).
    delta_e : float
             elevator deflection (rad).
    ih : float
         stabilator deflection (rad).
    delta_ail : float
               aileron deflection (rad).
    delta_r : float
             rudder deflection (rad).
             
    Returns
    -------
    
    forces : array_like
             3 dimensional vector with (F_x_s, F_y_s, F_z_s) forces in stability axes.
        
    References
    ----------
    AIRCRAFT DYNAMICS From modelling to simulation (Marcello R. Napolitano)
    chapter 3 and 4 
    """
 
    Long_coef_matrix = Long_Aero_Coefficients()
    Lat_coef_matrix = Lat_Aero_Coefficients()
      
    C_L_0, C_L_a, C_L_de, C_L_dih = Long_coef_matrix[0,:]
    C_D_0, C_D_a, C_D_de, C_D_dih = Long_coef_matrix[1,:]
    C_Y_b, C_Y_da, C_Y_dr = Lat_coef_matrix[0,:]
    
    C_L_full = C_L_0 + C_L_a * alpha + C_L_de * delta_e + C_L_dih * ih
    C_D_full = C_D_0 + C_D_a * alpha + C_D_de * delta_e + C_D_dih * ih  
    C_Y_full = C_Y_b * beta + C_Y_da* delta_ail + C_Y_dr * delta_r
    
    b = Geometric_Data()[2]
    c = Geometric_Data()[1]
    Sw = Geometric_Data()[0]
 
    aerodynamic_forces = q(U,rho) * Sw * np.array([-C_D_full, C_Y_full, -C_L_full])   #N
    
    return aerodynamic_forces
                         
def get_aerodynamic_moments( U, rho, alpha, beta, delta_e, ih, delta_ail, delta_r):
    
    """ Calculates forces 
    
    Parameters
    ----------
    
    rho : float
          density (kg/m3)
    U : float
        velocity (m/s)
    
    alpha : float
            attack angle (rad).
    beta : float
           sideslip angle (rad).
    delta_e : float
             elevator deflection (rad).
    ih : float
         stabilator deflection (rad).
    delta_ail : float
               aileron deflection (rad).
    delta_r : float
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

    
    C_m_0, C_m_a, C_m_de, C_m_dih = Long_coef_matrix[2,:]
    C_l_b, C_l_da, C_l_dr = Lat_coef_matrix[1,:]
    C_n_b, C_n_da, C_n_dr = Lat_coef_matrix[2,:]
    
    
    C_m_full = C_m_0 + C_m_a * alpha + C_m_de * delta_e + C_m_dih * ih
    C_l_full = C_l_b * beta + C_l_da* delta_ail + C_l_dr * delta_r
    C_n_full = C_n_b * beta + C_n_da* delta_ail + C_n_dr * delta_r 
    
    b = Geometric_Data()[2]
    c = Geometric_Data()[1]
    Sw = Geometric_Data()[0]

    aerodynamic_moments = q(U,rho) * Sw * np.array([C_l_full * b,C_m_full * c,C_n_full * b]) 
    
    return aerodynamic_moments            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    

