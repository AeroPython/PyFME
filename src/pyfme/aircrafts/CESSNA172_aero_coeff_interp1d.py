"""

Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Tests with Scipy interpolate interp1d for the 1-D coefficients

@author: andres.quezada.reed@gmail.com
@AeroPython

"""

import numpy as np 
import time
from scipy import interpolate

##############
### INPUTS ###
##############

b = 10.91								# [m]
c = 1.52								# [m]
V = 63									# [m/s]
alpha = 0								# [rad]
alpha_DEG = alpha * 57.29578			# [degree]
alpha_dot = 0 							# [rad/s]
beta = -0.085							# [rad]
delta_elev = 0							# [degree]
delta_aile = 0							# [degree]
delta_rud = 0							# [degree]
delta_rud_RAD = delta_rud * 0.0174533	# [rad] 
p = 0 									# [rad/s]
q = 0 									# [rad/s]
r = 0			 						# [rad/s]

##################
### TIMED PART ###
##################

start = time.time()

# LOADING OF DATA -- paleolithic style

alpha_data = np.array([-7.5,-5,-2.5,0,2.5,5,7.5,10,15,17,18,19.5]) # degree
delta_elev_data = np.array([-26,-20,-10,-5,0,7.5,15,22.5,28]) # degree
delta_aile_data = np.array([20,15,10,5,0,-2.5,-5,-10,-15]) # degree

CD_data = np.array([0.044,0.034,0.03,0.03,0.036,0.048,0.067,0.093,0.15,0.169,0.177,0.184])

CLsus_data = np.array([-0.571,-0.321,-0.083,0.148,0.392,0.65,0.918,1.195,1.659,1.789,1.84,1.889])
CLsus_alphadot_data = np.array([2.434,2.362,2.253,2.209,2.178,2.149,2.069,1.855,1.185,0.8333,0.6394,0.4971])
CLsus_q_data = np.array([7.282,7.282,7.282,7.282,7.282,7.282,7.282,7.282,7.282,7.282,7.282,7.282])
CLsus_delta_elev_data = np.array([-0.132,-0.123,-0.082,-0.041,0,0.061,0.116,0.124,0.137])

CY_beta_data = np.array([-0.268,-0.268,-0.268,-0.268,-0.268,-0.268,-0.268,-0.268,-0.268,-0.268,-0.268,-0.268])
CY_p_data = np.array([-0.032,-0.0372,-0.0418,-0.0463,-0.051,-0.0563,-0.0617,-0.068,-0.0783,-0.0812,-0.0824,-0.083])
CY_r_data = np.array([0.2018,0.2054,0.2087,0.2115,0.2139,0.2159,0.2175,0.2187,0.2198,0.2198,0.2196,0.2194])
CY_delta_rud_data = np.array([0.561,0.561,0.561,0.561,0.561,0.561,0.561,0.561,0.561,0.561,0.561,0.561])

Cl_beta_data = np.array([-0.178,-0.186,-0.1943,-0.202,-0.2103,-0.219,-0.2283,-0.2376,-0.2516,-0.255,-0.256,-0.257])
Cl_p_data = np.array([-0.4968,-0.4678,-0.4489,-0.4595,0.487,-0.5085,-0.5231,-0.4916,-0.301,-0.203,-0.1498,-0.0671])
Cl_r_data = np.array([-0.09675,-0.05245,-0.01087,0.02986,0.07342,0.1193,0.1667,0.2152,0.2909,0.3086,0.3146,0.3197])
Cl_delta_rud_data = np.array([0.0911,0.0818,0.0723,0.0627,0.053,0.0432,0.0333,0.0233,0.0033,-0.0047,-0.009,-0.015])
Cl_delta_aile_data = np.array([-0.078052,0.059926,-0.036422,-0.018211,0,0.018211,0.036422,0.059926,0.078052])

CM_data = np.array([0.0597,0.0498,0.0314,0.0075,-0.0248,0.068,-0.1227,-0.1927,-0.3779,-0.4605,-0.5043,-0.5496,])
CM_q_data = np.array([-6.232,-6.232,-6.232,-6.232,-6.232,-6.232,-6.232,-6.232,-6.232,-6.232,-6.232,-6.232])
CM_alphadot_data = np.array([-6.64,-6.441,-6.146,-6.025,-5.942,-5.861,-5.644,-5.059,-3.233,-2.273,-1.744,-1.356])
CM_delta_aile_data = np.array([0.3302,0.3065,0.2014,0.1007,-0.0002,-0.1511,-0.2863,-0.3109,-0.345])

CN_beta_data = np.array([0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0126])
CN_p_data = np.array([0.03,0.016,0.00262,-0.0108,-0.0245,-0.0385,-0.0528,-0.0708,-0.113,-0.1284,-0.1356,-0.1422])
CN_r_data = np.array([-0.028,-0.027,-0.027,-0.0275,-0.0293,-0.0325,-0.037,-0.043,-0.05484,-0.058,-0.0592,-0.06015])
CN_delta_rud_data = np.array([-0.2113,-0.215,-0.218,-0.22134,-0.2239,-0.226,-0.228,-0.229,-0.23,-0.23,-0.23,-0.23])

# INTERPOLATIONS

CD_interp = interpolate.interp1d(alpha_data, CD_data)

CLsus_interp = interpolate.interp1d(alpha_data, CLsus_data)
CLsus_alphadot_interp = interpolate.interp1d(alpha_data, CLsus_alphadot_data)
CLsus_q_interp = interpolate.interp1d(alpha_data, CLsus_q_data)
CLsus_delta_elev_interp = interpolate.interp1d(delta_elev_data, CLsus_delta_elev_data)

CY_beta_interp = interpolate.interp1d(alpha_data, CY_beta_data)
CY_p_interp = interpolate.interp1d(alpha_data, CY_p_data)
CY_r_interp = interpolate.interp1d(alpha_data, CY_r_data)
CY_delta_rud_interp = interpolate.interp1d(alpha_data, CY_delta_rud_data)

Cl_beta_interp = interpolate.interp1d(alpha_data, Cl_beta_data)
Cl_p_interp = interpolate.interp1d(alpha_data, Cl_p_data)
Cl_r_interp = interpolate.interp1d(alpha_data, Cl_r_data)
Cl_delta_rud_interp = interpolate.interp1d(alpha_data, Cl_delta_rud_data)
Cl_delta_aile_interp = interpolate.interp1d(delta_aile_data, Cl_delta_aile_data)

CM_interp = interpolate.interp1d(alpha_data, CM_data)
CM_q_interp = interpolate.interp1d(alpha_data, CM_q_data)
CM_alphadot_interp = interpolate.interp1d(alpha_data, CM_alphadot_data)
CM_delta_aile_interp = interpolate.interp1d(delta_elev_data, CM_delta_aile_data)

CN_beta_interp = interpolate.interp1d(alpha_data, CN_beta_data)
CN_p_interp = interpolate.interp1d(alpha_data, CN_p_data)
CN_r_interp = interpolate.interp1d(alpha_data, CN_r_data)
CN_delta_rud_interp = interpolate.interp1d(alpha_data, CN_delta_rud_data)

# CALCULATIONS

CD = CD_interp(alpha_DEG) # TBConsidered: CD_delta_elev_interp(alpha, delta_elev)

CLsus = CLsus_interp(alpha_DEG) + CLsus_delta_elev_interp(delta_elev) + (c/(2*V))*(CLsus_alphadot_interp(alpha_DEG)*alpha_dot + CLsus_q_interp(alpha_DEG)*q)

CY = CY_beta_interp(alpha_DEG)*beta + CY_delta_rud_interp(alpha_DEG)*delta_rud_RAD + (b/(2*V))*(CY_p_interp(alpha_DEG)*p + CY_r_interp(alpha_DEG)*r)

Cl = Cl_beta_interp(alpha_DEG)*beta + Cl_delta_aile_interp(delta_aile) + Cl_delta_rud_interp(alpha_DEG)*delta_rud_RAD + (b/(2*V))*(Cl_p_interp(alpha_DEG)*p + Cl_r_interp(alpha_DEG)*r)

CM = CM_interp(alpha_DEG) + CM_delta_aile_interp(delta_elev) + (c/(2*V))*(CM_q_interp(alpha_DEG)*q + CM_alphadot_interp(alpha_DEG)*alpha_dot) 

CN = CN_beta_interp(alpha_DEG)*beta + CN_delta_rud_interp(alpha_DEG)*delta_rud_RAD + (b/(2*V))*(CN_p_interp(alpha_DEG)*p + CN_r_interp(alpha_DEG)*r)  # TBConsidered: CN_delta_aile_interp(alpha, delta_aile)



end = time.time()
print("CD = ",CD,"CL = ",CLsus,"CY = ",CY)
print("Cl = ",Cl,"CM = ",CM,"CN = ",CN)
print(end - start)


