import numpy as np
from numpy import linalg as LA
from model import parameters as param
from numpy.random import random
from scipy import linalg


#------- allocations of variables ---------

s_DB = np.zeros((param.nstate,param.nstate)) # D and B overlap
s_BA = np.zeros((param.nstate,param.nstate)) # B and A overlap
s_DA = np.zeros((param.nstate,param.nstate)) # D and A overlap

#---------------- defining delta functions------------

def delta(i,j):
    if i==j:
        return 1.0
    else:
        return 0.0


#----------------- overlap terms ----------------------

def getoverlap(R_0,param):
  

    omega_c = param.omega_c
    nstate = param.nstate
    
    s = np.zeros((nstate,nstate))
    sprime = np.zeros((nstate,nstate))

    for i in range(nstate):
        for j in range(nstate):

            s[i,j] = (i+0.5)*(omega_c)*delta(i,j) + 0.5*((omega_c)**2)*(R_0**2)*delta(i,j) \
                        -(((omega_c**1.5)*R_0)/(np.sqrt(2.0)))*(np.sqrt(np.real(j+1))*delta(i,j+1) \
                             + np.sqrt(np.real(j))*delta(i,j-1)) 

    eigen, sprime = np.linalg.eigh(s) 

    return sprime

# =========== rate constant calculations =========================

#--------- calculation of V_DB (with channel n,l)---------------
vdb = np.zeros((param.nfock,param.nbridge,param.nstep)) #light induced DB coupling

for g in range(param.nstep):
    xi = g*3.6749303600696764*10**-6 + 10**-20

    R0_DB = np.sqrt(2.0/(param.omega_c)**3)* xi * param.del_mu_DB 

    s_DB = getoverlap(R0_DB,param) # D and B overlaps

    for n in range(param.nfock):
        for l in range(param.nbridge):

            vdb[n,l,g] = param.diab_DB*s_DB[n,l] \
                + xi*(param.mu_DB)*np.sqrt(np.real(l)+1)*(s_DB[n,l+1]) \
                + xi*(param.mu_DB)*np.sqrt(np.real(l))*(s_DB[n,l-1])


#------ calculation of V_BA (with channel l,m)-----------------------
vba = np.zeros((param.nbridge,param.nfock,param.nstep)) # light induced BA coupling

for g in range(param.nstep):
    xi = g*3.6749303600696764*10**-6 + 10**-20

    R0_BA = np.sqrt(2.0/(param.omega_c)**3)* xi * param.del_mu_BA

    s_BA = getoverlap(R0_BA,param) # B and A overlaps


    for l in range(param.nbridge):
        for m in range(param.nfock):

            vba[l,m,g] = param.diab_BA*s_BA[l,m] \
                + xi*(param.mu_BA)*np.sqrt(np.real(m)+1)*(s_DB[l,m+1]) \
                + xi*(param.mu_BA)*np.sqrt(np.real(m))*(s_DB[l,m-1])

#------ calculation of V_DA coupling term (for each n,m channels)------------
vda = np.zeros((param.nfock,param.nfock,param.nstep)) # light induced DA coupling

for g in range(param.nstep):
    xi = g*3.6749303600696764*10**-6 + 10**-20

    R0_DA = np.sqrt(2.0/(param.omega_c)**3)* xi * param.del_mu_DA

    s_DA = getoverlap(R0_DA,param) # D and A overlaps

   
    for n in range(param.nfock):
        for m in range(param.nfock):
            for l in range(param.nbridge):
                vda[n,m,g] = vda[n,m,g] - 0.5*vdb[n,l,g]*vba[l,m,g]*((1.0/((param.dE+param.bias)+(l-m)*param.omega_c)) \
                    + (1.0/(param.dE+(l-n)*param.omega_c)))



    for n in range(param.nfock):
        for m in range(param.nfock):
            vda[n,m,g] = vda[n,m,g] \
                + (((xi)**2 * param.mu_DB * param.mu_BA)/param.omega_c)*s_DA[n,m] 


#----- prefactor for the rate calculation (for each channels) ---------------

A = np.zeros((param.nfock,param.nfock,param.nstep)) #total prefactor term

for g in range(param.nstep):

    xi = g*3.6749303600696764*10**-6 + 10**-20

    for n in range(param.nfock):
        for m in range(param.nfock):

            A[n,m,g] = 2.0*param.pi*(vda[n,m,g])**2*(1.0/np.sqrt(4.0*param.pi*param.lam*(1.0/param.beta)))
            A[n,m,g] = A[n,m,g]*41341.37


#------------- calculation of delg_nm term------------------------------

delg = np.zeros((param.nfock,param.nfock,param.nstep))

for g in range(param.nstep):

    xi = g*3.6749303600696764*10**-6 + 10**-20

    for n in range(param.nfock):
        for m in range(param.nfock):
            for l in range(param.nbridge):
                delg[n,m,g] = delg[n,m,g] + vdb[n,l,g]*np.transpose(vdb[m,l,g])*(1.0/(param.dE+(l-n)*param.omega_c)) \
                    - np.transpose(vba[l,n,g])*vba[l,m,g]*(1.0/((param.dE+param.bias)+(l-m)*param.omega_c))



#------- rate for each channel with varygin Donor to acceptor energy gap--------
k = np.zeros((param.nfock,param.nfock,param.nstep)) #rate for each channel

for g in range(param.nstep):
    xi = g*3.6749303600696764*10**-6 + 10**-20

    for n in range(param.nfock):
        for m in range(param.nfock):
            k[n,m,g] = A[n,m,g]*np.exp(-(-param.bias+delg[n,m,g]+param.lam - (np.real(n)*param.omega_c) \
                + (np.real(m)*param.omega_c))**2/(4.0*param.lam*(1.0/param.beta))) # in (ps)^-1 

#-------- total partition function ---------------------

part = 0

for n in range(param.nfock):
    part  = part + np.exp(-param.beta*np.real(n)*param.omega_c)

#--------- calculating net rate -------------------------
f = open("rate_vs_gc_withbias_0.15_omega40mev.txt","w+")

# this will be the total rate, array of energy-scan
total_rate = np.zeros(param.nstep)

for g in range(param.nstep):

    xi =  g*3.6749303600696764*10**-6 + 10**-20

    for n in range(param.nfock):
        for m in range(param.nfock):

            total_rate[g] = total_rate[g] + k[n,m,g]*(np.exp(-param.beta*np.real(n)*param.omega_c)/part)


    f.write(f"{((xi*27.2114*1000.0)/(param.omega_c*27.2114*1000.0))} {total_rate[g]} \n")        
f.close()
