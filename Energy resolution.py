#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:55:30 2023

@author: Sophie Lake, Noel Shipley
"""

import numpy as np
import matplotlib.pyplot as plt

kaon_m = 0.493677 #kaon mass in GeV with an error of 0.013 Mev
muon_m = 0.105658 #muon mass in GeV with an error of 0.0000023 Mev

kaon_p = 75 #kaon momentum value with error of 1.6%
kaon_E = np.sqrt(kaon_m**2 + kaon_p**2)

gtk_res = 0.002*kaon_p #momentum resolution of GTK
#straw_res = 0.003000416638*(muon_p**2)

kaon_res = gtk_res * kaon_p/kaon_E
#muon_res = straw_res*muon_p/(np.sqrt(muon_p**2 + muon_m**2))

#neutrino_E_res = np.sqrt(kaon_res**2 + muon_res**2)

neutrino_E = np.loadtxt("neutrino_energies.txt")
neutrino_E = np.round(neutrino_E,3)
neutrino_E = np.unique(neutrino_E, return_counts=False)

frac = neutrino_E /kaon_E

muon_E = kaon_E - neutrino_E  #an array of the muon energies 

neutrino_res = np.sqrt(kaon_res**2 + ((0.003000416638*(muon_E**2 - muon_m**2)**(3/2))/(muon_E))**2)/neutrino_E  #an array of sigma_E/E of the neutrino energies 

x_values = neutrino_E
y_values = neutrino_res
log = np.log(y_values)
x = np.linspace(min(x_values),max(x_values),100)

fit = np.polyfit(x_values,log,1) #linear fit for log plot
fit = np.flip(fit)

plt.figure()
plt.title("Logarithmic Energy resolution on the tagged neutrino")
plt.plot(x_values,log, 'b', label = "Uncertainty in NA62")
plt.plot(x, fit[0] + fit[1]*x, 'k', label = "Linear fit")
plt.xlabel('Energy of the neutrinos (GeV)')
plt.ylabel('log($\sigma$E/E)')
plt.legend()
plt.show()

plt.figure()
plt.title("Energy resolution on the tagged neutrino")
plt.plot(x_values, y_values, label ="Uncertainty in NA62")
plt.plot(x, np.exp(fit[0]) * np.exp(fit[1]*x), 'k', label = "Exponential fit")
plt.xlabel('Energy of the neutrinos (GeV)')
plt.ylabel('$\sigma$E/E')
plt.legend()
plt.show()

