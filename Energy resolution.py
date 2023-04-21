#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:55:30 2023

@author: Sophie Lake, Noel Shipley
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import pareto

'''
Errors on tagged neutrino as a function of energy
'''

kaon_m = 0.493677 #kaon mass in GeV with an error of 0.013 Mev
muon_m = 0.105658 #muon mass in GeV with an error of 0.0000023 Mev

kaon_p = 75 #kaon momentum value with error of 1.6%
kaon_E = np.sqrt(kaon_m**2 + kaon_p**2)

kaon_res_p = 0.002*kaon_p #momentum resolution of GTK

neutrino_E = np.loadtxt("neutrino_energy_dist.txt")
neutrino_E = np.round(neutrino_E,3)
neutrino_E = np.unique(neutrino_E, return_counts=False)

muon_E = kaon_E - neutrino_E  #an array of the muon energies 
muon_p = np.sqrt(muon_E**2-muon_m**2)
muon_res_p = np.sqrt(0.003**2+(0.00005*muon_p)**2)*muon_p

neutrino_res = np.sqrt((kaon_p*kaon_res_p/kaon_E)**2+(muon_p*muon_res_p/muon_E)**2)/neutrino_E

x_values = neutrino_E
y_values = neutrino_res
x = np.linspace(np.min(x_values),np.max(x_values),100)

popt, pcov = opt.curve_fit(lambda x,a,b,c: a/x**b + c, x_values, y_values)

plt.figure()
plt.title("Energy resolution on the tagged neutrino")
plt.plot(x_values, y_values, label ="Uncertainty in NA62")
plt.plot(x,popt[0]/x**popt[1] + popt[2],'k--',label = "Power law fit")
plt.xlabel('Energy of the neutrinos (GeV)')
plt.ylabel('$\delta$(E)/E')
plt.legend()
plt.show()

print(f"The fractional error on the tagged neutrinos has the form a/E**b + c where [a,b,c] are {popt} respectively")

'''
Errors on cross-section from MINOS data as a function of energy
'''

x = np.linspace(3,60,100)
x2 = np.array([3.48,4.45,5.89,7.97,10.45,13.43,16.42,19.87,23.88,27.89,32.81,38.87,45.77])#MINOS Energy values
y2 = np.array([0.748,0.711,0.708,0.722,0.699,0.691,0.708,0.689,0.683,0.686,0.675,0.675,0.676])#MINOS cross-section values
y2_err = np.array([0.061,0.033,0.032,0.045,0.043,0.028,0.020,0.016,0.015,0.016,0.016,0.018,0.019])#MINOS cross-section errors

err = y2_err/y2 #fractional error

b = 0.3
rv = pareto(b)  #Fitting inverse power law to errors

plt.figure()
plt.plot(x2,err,'bo',label = "Fractional errors") #MINOS erros
plt.plot(x, rv.pdf(x) + np.min(err), 'k-', lw=2, label = 'Power law fit') #Trend line
plt.legend()
plt.title(r"Errors from MINOS for ${}^{56} \rm Fe$")
plt.ylabel(r'Fractional error of ${}^{56} \rm Fe$')
plt.xlabel('Energy (GeV)')
plt.show()

print(f"The fractional error on the cross-section values has the form {b}*E**-{b} + {np.min(err)}")
