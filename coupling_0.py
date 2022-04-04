import numpy as np
from numpy.random import random

#------------ parameters all in au------
nstep = 50
pi  = 3.1415926
lam = 0.65/27.2114
diab_DB = 0.005/27.2114
diab_BA = 0.005/27.2114
beta = 1052.85
bias = 0.15/27.2114
#----------------------------------------

#------ effective DA couplings -----------

vda_eff = np.zeros(nstep)

for g in range(nstep):
    dg = g*0.018374 + 10**-10

    vda_eff[g] = - (0.5*diab_DB*diab_BA)*((1.0/(dg+bias)) + (1.0/dg))

#------- prefactor for reaction rate ----------

A = np.zeros(nstep)

for g in range(nstep):
    dg = g*0.018374 + 10**-10
    A[g] = 2.0*pi*(vda_eff[g])**2*(1.0/np.sqrt(4.0*pi*lam*(1.0/beta)))
    A[g] = A[g]*41341.37


#========== total rate ===============

f = open("rate_vs_delE_0_coup_withbias_0.txt","w+")
total_rate = np.zeros(nstep)

for g in range(nstep):
    dg = g*0.018374 + 10**-10

    delg = bias*((diab_BA**2/((dg+bias)*dg))-1.0) 

    total_rate[g] = A[g]*np.exp(-(delg+lam)**2/(4.0*lam*(1.0/beta)))

    f.write(f"{dg*27.2114} {total_rate[g]} \n")    

f.close()