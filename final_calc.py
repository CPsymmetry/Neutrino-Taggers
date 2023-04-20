# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:43:11 2023

@author: Noel Shipley
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def Integral(L_mat,xsec,trig,Errors=False):
    '''
    The rate of interaction:
         W = integral(L_mat*xsec*E_dist*LO_dist*beamuptime*trigger*MUV3_eff dE)
         returns W in units of per second
         
    Errors on W
        factor = sqrt(sum of fractional uncertainties squared)
        delta W = integral(L_mat*xsec*E_dist*L0_dist*beam_uptime*trigger*MUV3_eff*factor dE)
    '''
    if Errors == False:
        I = lambda E: L_mat*np.polyval(xsec[::-1],E)*1e-42*np.polyval(pol,E)*np.polyval(L0[::-1],E)*beam_uptime*trig*muv_eff
    else:
        I = lambda E: L_mat*np.polyval(xsec[::-1],E)*1e-42*np.polyval(pol,E)*np.polyval(L0[::-1],E)*beam_uptime*trig*muv_eff*np.sqrt((0.11/63.56)**2 + (s_len/(A*d_len)*(1-A)*1.2/75)**2 + (0.005/geometric)**2 +  (1.02680571*E**(-1.43348205) -3.39359323e-04)**2 + (0.3*E**(-1.3)+0.02196)**2)
    return I

neutrino_E = np.loadtxt("neutrino_energy_dist.txt")
neutrino_E = np.round(neutrino_E,1)
scale = 3*np.arange(21)  #for x-axis scale
ax,ay = np.unique(neutrino_E, return_counts=True)   #finds unique values, returns them and number of occurences
ay = ay/np.sum(ay) #for numerical analysis
x_new = np.linspace(np.min(ax),np.max(ax),1000)
size = 15
pol = np.polyfit(ax,ay,size)  #gets coefficients of polynomial in decreasing order
np.savetxt('nu.txt', pol[::-1]) #writes terms to file in increasing order  i.e. [x**0,x**1,x**2...]
n = lambda E: np.polyval(pol,E) #np.polyval requires the function to be writen as a*x**n + b*x**(n-1) ... + w*x + z
norm = integrate.quad(n,np.min(ax),np.max(ax))   #area under weight function has to be equal 1    

plt.figure()
plt.plot(ax,ay/norm[0],label = "Simulated data")        #simulated data
plt.plot(x_new,np.polyval(pol,x_new)/norm[0],'k', label = f"polynomial order {size} fit")  #polynomial fit
plt.legend()
plt.xticks(scale)
plt.title("Normalised neutrino energy distribution")
plt.ylabel('Fractional number of occurences')
plt.xlabel('Energy (Gev)')
plt.show()

geometric = 0.215    #geometric acceptance
k_initial = 45e6     #initial kaon flux
u = 1.66054e-27      #atomic mass unit
d_len = 563.780      #decay length of kaon with 75GeV/cz
s_len = 80           #length to straw from CHANTI
A = 1-np.exp(-s_len/d_len)
phi = k_initial*A*geometric*0.6356*0.982     #rate of nuetrinos in geometric acceptance
L_lkr = 2413*1.27/(83.798*u)*phi      #L_mat in LKr only
L_muv = 7874*1.3/(55.845*u)*phi       #L_mat in MUV1 and MUV2
L_filter = 7874*0.8/(55.845*u)*phi    #L_mat in Iron Filter only
lkr_trig = 0.2366           #fraction that deposit at least 5GEV in LKr
iron_trig = 0.7259        #fraction that deposit at least 5GEV in MUV 1,2
muv_eff = 0.9852692515123929
year = 200*24*60*60  #a 200 day year 
beam_uptime = 9000/86400 #proprtion of time the beam is on per day

lkr_abun = np.array([0.5699, 0.1728, 0.1159, 0.1150]) #weighting the cross-sections of natural Krypton
kr_82 = np.array([37.7050, 58.6816, -0.02038, 0.0000732]) 
kr_83 = np.array([38.3111, 59.6001, -0.02091, 0.0000751])
kr_84 = np.array([38.9045, 60.5194, -0.02145, 0.0000710])
kr_86 = np.array([40.1076, 62.3570, -0.02252, 0.0000811])
lkr_xsec = (lkr_abun[0]*kr_84 + lkr_abun[1]*kr_86 + lkr_abun[2]*kr_82 + lkr_abun[3]*kr_83)/np.sum(lkr_abun) #coefficients of cross-section in increasing order

iron_abun = np.array([0.9175, 0.0585,0.0212])
Fe_54 = np.array([22.794, 37.64440, -0.0130259, 0.0000486])
Fe_56 = np.array([27.314, 39.38812, -0.0111567, 0.0000288])
Fe_57 = np.array([24.458, 40.40750, -0.0147505, 0.0000552])
iron_xsec = (iron_abun[0]*Fe_56 + iron_abun[1]*Fe_54 + iron_abun[2]*Fe_57)/np.sum(iron_abun) #coefficients of cross-section in increasing order

L0 = np.array([1.0418242853821924,-0.004143528018613048])   #coefficients of L0 efficiency in increasing order

lkr_int = integrate.quad(Integral(L_lkr,lkr_xsec,lkr_trig,Errors=False),np.min(ax),np.max(ax))#get value
lkr_err = integrate.quad(Integral(L_lkr,lkr_xsec,lkr_trig,Errors=True),np.min(ax),np.max(ax))#get error

iron_muv_int = integrate.quad(Integral(L_muv,iron_xsec,iron_trig,Errors=False),np.min(ax),np.max(ax))
iron_muv_err = integrate.quad(Integral(L_muv,iron_xsec,iron_trig,Errors=True),np.min(ax),np.max(ax))

iron_filter_int = integrate.quad(Integral(L_filter,iron_xsec,1,Errors=False),np.min(ax),np.max(ax))
iron_filter_err = integrate.quad(Integral(L_filter,iron_xsec,1,Errors=True),np.min(ax),np.max(ax))

print(f"The number of tagged neutrinos per year in LKr only with 5GEV trigger conditions is {lkr_int[0]*year/norm[0]} with an uncertainty of {lkr_err[0]*year/norm[0]}")
print(" ")
print(f"The number of tagged neutrinos per year in LKr and MUV 1,2 with 5GEV trigger conditions is {(lkr_int[0] + iron_muv_int[0])*year/norm[0]} with an uncertainty of {(lkr_err[0]+iron_muv_err[0])*year/norm[0]}")
print(" ")
print(f"The number of tagged neutrinos per year in LKr, MUV 1,2 and Iron filter with no trigger conditions is {(lkr_int[0]/lkr_trig + iron_muv_int[0]/iron_trig + iron_filter_int[0])*year/norm[0]} with an uncertainty of {(lkr_err[0]/lkr_trig + iron_muv_err[0]/iron_trig + iron_filter_err[0])*year/norm[0]}")