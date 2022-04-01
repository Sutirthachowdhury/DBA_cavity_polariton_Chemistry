import numpy as np
import math
import random

class parameters():
    nstate = 100 # number of oxillary states req for diagonalization
    nfock = 20   # number of fock states
    nbridge = 20 # number of virtual bridge states, here we have one bridge
    nstep = 50 # energy scan steps
    pi = 3.1415926  # value of pi
    diab_DB = 0.005/27.2114  # diabatic coupling betwn D and B
    diab_BA = 0.005/27.2114  # diabatic coupling betwn B and A
    xi = 0.001/27.2114  # light-matter cooupling strength
    mu_DB = 1.0 # transition dipole betwn D and B
    mu_BA = 1.0 # transition dipole betwn B and A
    lam = 0.65/27.2114 # solvent reorganization energy 
    omega_c = 0.2/27.2114 # photon frequency
    beta = 1052.85 # temperature of the reaction
    del_mu_DA = 1.0 # del_mu_DA = mu_DD - mu_AA (permanent dipole diff)
    del_mu_DB = 5.0 # del_mu_DB = mu_DD - mu_BB (permanent dipole diff)
    del_mu_BA = 5.0 # del_mu_BA = mu_BB - mu_AA (permanent dipole diff)