#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:49:11 2022

@author: hedinyer
"""

#####################################################
# import libraries
#####################################################


#non-importan libs
#####################################
import numpy as np                    #-> turn python into a "matlab"
import matplotlib.pyplot as plt       #-> plot lib            
from scipy.signal import find_peaks   #-> very importan to find peaks in COM signal
import math                           #-> mathematical tool
import skrf as rf                     #-> important library to load .s2p data
#####################################


#optimization libs
#####################################
from scipy.integrate import odeint
import scipy.optimize as ot
from scipy.optimize import minimize
from scipy import optimize
#####################################

#to ignore warnings
#####################################
import warnings
warnings.filterwarnings('ignore')
#####################################

######################################################
######################################################

#Global stuff
########################################
global geo
global x1
global cut_array
global cut_f
global freq_vect
global ntwk
global s_r
global s_i
global s_t
global peak
global z
global f
global n
global n


#1-import the data into ntwk variable
#####################################

ntwk = rf.Network('/home/hedinyer/Documents/python_germany/optimization/optinization_1/Com_Optimization_py/P450U_W1_WR1_WC1_SR46_SC1.S1P')

#2-and now with the loaded data it's possible to 
#  asign it into multiple variables like the frecuency and 
#  the real and imag values
#####################################

s_r = ntwk.y.real      #<--- only real values
s_i = ntwk.y.imag      #<--- only imag values
s_t = ntwk.y           #<--- the real and the imag values

#3-use reshape to fix the matrix to work with them
#####################################

s_r = s_r.reshape(-1)  
s_i = s_i.reshape(-1)
s_t = s_t.reshape(-1)
#3.1-turn into numpy array
##########################

s_r = np.array(s_r)
s_i = np.array(s_i)
s_t = np.array(s_t)

#4-get the frecuency
#####################################

f = ntwk.f


#6-find peaks of signal
######################################

peak, z = find_peaks(s_t, height=0.011) #<--- with height you can take the desired peak
n = z['peak_heights']
m = f[peak]

#7-cut the region of interes
#######################################


#7.1-find nearest function
##########################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
##########################



def Ultra_Optim(j):
    upper_region_array =  m[0]*j[0]/100 + m[0]
    lower_region_array = -m[0]*j[1]/100 + m[0]

    upper_region_array = find_nearest(f, value=upper_region_array)
    lower_region_array = find_nearest(f, value=lower_region_array)

    upper_region_array = np.where(f == upper_region_array)
    lower_region_array = np.where(f == lower_region_array)

    upper_region_array = list(upper_region_array[0])
    lower_region_array = list(lower_region_array[0])

    upper_region_array = upper_region_array[0]
    lower_region_array = lower_region_array[0]

    iterion_numer = upper_region_array - lower_region_array
    int_iterarion_number = int(iterion_numer)
    int_lower_region_array = int(lower_region_array)

    cut_array = []
    cut_f = []

    cut_array2 = []
    cut_f2 = []
    
    #7.2-to total region
    ##########################

    for i in range (int_iterarion_number):
        cut_array.append(s_t[i + int_lower_region_array])
        cut_f.append(f[i + int_lower_region_array])

    #7.3-to imag region
    ##########################

    for i in range (int_iterarion_number):
        cut_array2.append(s_i[i + int_lower_region_array])
        cut_f2.append(f[i + int_lower_region_array])
        
        
    #7.4-plot the cut region (total)
    ##########################

    cut_f = np.array(cut_f)
    cut_array = np.array(cut_array)
    
    
    cut_f2 = np.array(cut_f2)
    cut_array2 = np.array(cut_array2)
    
    
    #8-define the geometry of com model
    ###########################

    geom_p        = 1.05e-6
    geom_numf     = 401
    geom_numgra   = 25
    geom_aperture = 15.75
    freq_vect = cut_f
    geo = [geom_p, geom_numf, geom_numgra, geom_aperture]
    
    
    #9-define the model
    ###########################

    def Com_Model(x):
        M_const = 1
        atan_squishFactor = 10
        pi = math.pi
        K11        = 0;
        K12        = x[0];
        Zeta       = x[1];
        Gamma_1    = x[2];
        Gamma_2    = x[3];
        Vs         = x[4];
        R_parallel = x[5];
        C          = x[6];
        
        p= geo[0]
        
        ResonanceEstimate = (Vs*(-K12 + pi/p))/(2*pi)
        AntiresonanceEstimate = ((Vs*(K12 + pi/p))/(2*pi))
        
        # Precalculate 
        rfl_length = p*geo[2];
        rfl_idt_gap = 0;
        idt_length = p*geo[1];
        
        
        omega = 2 * pi * freq_vect
        vref =  Vs/ (1 - (K11 * p / (M_const * pi)))
        
        gamma_freq = 0.5 * (Gamma_1 + Gamma_2) +((Gamma_2 - Gamma_1)) / pi * np.arctan(atan_squishFactor * (freq_vect - (AntiresonanceEstimate + ResonanceEstimate) / 2) / (AntiresonanceEstimate - ResonanceEstimate))

        theta_u = omega / vref - M_const * (pi / p) + K11 - 1j * gamma_freq
        
        theta_p1 = -(np.sqrt((theta_u)**2) - abs(K12)**2)
        theta_p2 = 1j * (np.sqrt(abs(K12)**2 - (theta_u)**2))
        theta_p3 = np.sqrt((theta_u)**2 - abs(K12)**2)
        
        is_theta_p1_true = (theta_u < -abs(K12))
        is_theta_p2_true = (abs(theta_u) < abs(K12))
        
        is_theta_p1_true = is_theta_p1_true.astype(int)
        is_theta_p2_true = is_theta_p2_true.astype(int)
        
        
        is_theta_p3_true = np.ones(len(is_theta_p1_true)) - is_theta_p1_true - is_theta_p2_true

        theta_p = theta_p1 * is_theta_p1_true + theta_p2 * is_theta_p2_true + theta_p3 * is_theta_p3_true
        
        gamma_plus = (theta_p - theta_u) / K12

        gamma_zero = gamma_plus
     
        exprval = np.exp(-2j * theta_p * rfl_length)
        gamma = gamma_zero * ((1 - exprval) / (1 - (gamma_zero**2) * exprval))
        
        
        BETA = 1
        gamma_t = gamma * np.exp(-2j * BETA * rfl_idt_gap)

        P1 = 1j * Zeta**2 * idt_length / (theta_u + K12)

        P2 = 2 / (theta_p * idt_length)

        P3_r = 1/np.tan(theta_p * idt_length * 0.5);

        P3_i = ((1 + gamma_t) * (1 - gamma_zero)) / ((1 - gamma_t) * (1 + gamma_zero))

        P4 = 1j * omega * C * idt_length

        P3 = P3_r + 1j * P3_i
        
        # #  COM model admittance
        Y_no_parasite= (P1 * ((P2 * P3** - 1) -1) + P4); 
        Y_total = (Y_no_parasite**(-1) + R_parallel)**(-1);
        
        ares_Z,ares_idx = max(np.log(abs(1/cut_array))), np.argmax(np.log(abs(1/cut_array)))
        res_Y,res_idx   = max(np.log(abs(cut_array)))  , np.argmax(np.log(abs(cut_array)))
        
        YZ_offset = ares_Z - res_Y
        
        YZ_win = np.zeros(len(cut_array))
        YZ_win[round((res_idx + ares_idx)/2):round((3*ares_idx)/2 - res_idx/2)] = 1
        
        
        def hybrid_log_abs_YZ(Y_in, YZ_win, YZ_offset):
            result = (np.log(abs(Y_in))*(1 - YZ_win) + (np.log(abs(1/Y_in)) - YZ_offset)*YZ_win)
            return result
        
        
        valor1 = hybrid_log_abs_YZ(Y_total, YZ_win, YZ_offset) - hybrid_log_abs_YZ(cut_array, YZ_win, YZ_offset)
        valor2 = np.log(abs((Y_total.real))) - np.log(abs(cut_array.real))
        
        valor1[np.isnan(valor1)] = 0
        valor2[np.isnan(valor2)] = 0
        
        valor1 = valor1**2
        valor2 = valor2**2
        
        result = np.concatenate((valor2, valor1))
        result = sum(result)
        
        
        return result
    
    
    #init guess from matlab
    K11 = 2000;
    K12 = 120000;
    Zeta = 3500; 
    Gamma_1 = 3;
    Gamma_2 = 500;
    Vs = 3700;
    R_ser = 0.2;
    C = 1e-08;
    
    #second x0
    x0 = np.array([K12,Zeta,Gamma_1,Gamma_2,Vs,R_ser,C])
    # Bounds for the 2 principal models
    bnds = ((500, j[2]), (50, j[3]), (500,j[4]), (500,j[5]), (500,j[6]), (1e-03,j[7]) ,(1e-15,j[8]))

    res = minimize(Com_Model,x0,method='Nelder-Mead', tol=1e-6, bounds=(bnds))
    cost_fuc2_val = Com_Model(res.x)
    print(res.x,'\n')
    print(cost_fuc2_val,'\n')
    
    #9-define the model
    ###########################

    def Com_Model2(x):
        M_const = 1
        atan_squishFactor = 10
        pi = math.pi
        K11        = 0;
        K12        = x[0];
        Zeta       = x[1];
        Gamma_1    = x[2];
        Gamma_2    = x[3];
        Vs         = x[4];
        R_parallel = x[5];
        C          = x[6];
        
        p= geo[0]
        
        ResonanceEstimate = (Vs*(-K12 + pi/p))/(2*pi)
        AntiresonanceEstimate = ((Vs*(K12 + pi/p))/(2*pi))
        
        # Precalculate 
        rfl_length = p*geo[2];
        rfl_idt_gap = 0;
        idt_length = p*geo[1];
        
        
        omega = 2 * pi * freq_vect
        vref =  Vs/ (1 - (K11 * p / (M_const * pi)))
        
        gamma_freq = 0.5 * (Gamma_1 + Gamma_2) +((Gamma_2 - Gamma_1)) / pi * np.arctan(atan_squishFactor * (freq_vect - (AntiresonanceEstimate + ResonanceEstimate) / 2) / (AntiresonanceEstimate - ResonanceEstimate))

        theta_u = omega / vref - M_const * (pi / p) + K11 - 1j * gamma_freq
        
        theta_p1 = -(np.sqrt((theta_u)**2) - abs(K12)**2)
        theta_p2 = 1j * (np.sqrt(abs(K12)**2 - (theta_u)**2))
        theta_p3 = np.sqrt((theta_u)**2 - abs(K12)**2)
        
        is_theta_p1_true = (theta_u < -abs(K12))
        is_theta_p2_true = (abs(theta_u) < abs(K12))
        
        is_theta_p1_true = is_theta_p1_true.astype(int)
        is_theta_p2_true = is_theta_p2_true.astype(int)
        
        
        is_theta_p3_true = np.ones(len(is_theta_p1_true)) - is_theta_p1_true - is_theta_p2_true

        theta_p = theta_p1 * is_theta_p1_true + theta_p2 * is_theta_p2_true + theta_p3 * is_theta_p3_true
        
        gamma_plus = (theta_p - theta_u) / K12

        gamma_zero = gamma_plus
     
        exprval = np.exp(-2j * theta_p * rfl_length)
        gamma = gamma_zero * ((1 - exprval) / (1 - (gamma_zero**2) * exprval))
        
        
        BETA = 1
        gamma_t = gamma * np.exp(-2j * BETA * rfl_idt_gap)

        P1 = 1j * Zeta**2 * idt_length / (theta_u + K12)

        P2 = 2 / (theta_p * idt_length)

        P3_r = 1/np.tan(theta_p * idt_length * 0.5);

        P3_i = ((1 + gamma_t) * (1 - gamma_zero)) / ((1 - gamma_t) * (1 + gamma_zero))

        P4 = 1j * omega * C * idt_length

        P3 = P3_r + 1j * P3_i
        
        # #  COM model admittance
        Y_no_parasite= (P1 * ((P2 * P3** - 1) -1) + P4); 
        Y_total = (Y_no_parasite**(-1) + R_parallel)**(-1);
        return Y_total
    
    vector = Com_Model2(res.x)
    
    plt.plot(cut_f,cut_array.imag)
    plt.plot(cut_f,vector.imag,"--")
    plt.legend(["Exp","Sym"])
    plt.show()

    
    
    return cost_fuc2_val



x0=[5,5,1000,1000,1000,1000,1000,0.2,1e-5]
bnds=((1,12),(1,12),(0.1,5000),(0.1,5000),(0.1,5000),(0.1,5000),(0.1,5000),(0.1,0.4),(1e-15,1e-1))
res2 = minimize(Ultra_Optim,x0,method='Nelder-Mead', bounds=(bnds),options={'maxfev':50})
print(res2.x)

print("2")

bnds2=((1,12),(1,12),(0.1,5000),(0.1,5000),(0.1,5000),(0.1,5000),(0.1,5000),(0.1,0.4),(1e-15,1e-1))
res3 = minimize(Ultra_Optim,res2.x,method='Nelder-Mead', bounds=(bnds2),options={'maxfev':50})
print(res3.x)
