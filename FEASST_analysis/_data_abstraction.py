import os
import numpy as np
import feasst
import copy

from ._append_operations import *

def data_abstraction(source,prefix):

    #Determine restart file structure/pattern
    input_files = [ x for x in os.listdir(source) if (prefix in x and "criteria" in x and "bak" not in x and "rng" not in x) ]
    windows = len(input_files)
                
    # Stitch the windows together
    for window in range(windows):
        file_stub = source+prefix+str(window)
            
        space = feasst.Space(file_stub+"space")
        boxLength = [space.boxLength(0),space.boxLength(1),space.boxLength(2)]
        criteria = feasst.CriteriaWLTMMC(file_stub+"criteria")
        beta = criteria.beta()
        lnZ = np.log(criteria.activ(0))
        volume = boxLength[0]*boxLength[1]*boxLength[2]
        nMin = int(np.ceil(criteria.mMin()))
        nMax = int(np.floor(criteria.mMax()))
            
        N_w = [x for x in range(nMin,nMax+1)]
        lnPi_w = [x for x in np.array(criteria.lnPIdouble())]
        bins = len(lnPi_w)
        energy_w = [ criteria.pe(i).sumDble()/criteria.pe(i).nValues() if criteria.pe(i).nValues()!=0 else 0. for i in range(0,bins)]
        #energy2_w = [ criteria.pe(i).sumSqDble()/criteria.pe(i).nValues() if criteria.pe(i).nValues()!=0 else 0. for i in range(0,bins)]

        # Return FEASST data as Python lists to enable easier appending
        if window == 0:
            # Deep copy first window into master arrays
            N = copy.deepcopy(N_w)
            lnPi = copy.deepcopy(lnPi_w)
            energy = copy.deepcopy(energy_w)
            #energy2 = copy.deepcopy(energy2_w)
        else:
            # Append to master arrays
            #append_array(master=lnPi,newdata=lnPi_w,smooth_type="mean")
            Nold = [x for x in N]
            N, lnPi = append_data(N,lnPi,N_w,lnPi_w,smooth_type="mean")
            #Ntmp, energy = append_data(Nold,energy,N_w,energy_w,smooth_type="mean")
            Ntmp, energy = append_energy(Nold,energy,N_w,energy_w)
            #Ntmp, energy2 = append_data(Nold,energy2,N_w,lnPi_w,smooth_type="mean")


    #Convert to NumPy Arrays
    lnPi = np.array(lnPi)
    energy = np.array(energy)
    
    # Normalize lnPi
    lnPi = lnPi - max(lnPi)
    lnPi = lnPi - np.log(sum(np.exp(lnPi)))
    
    # Adjust Energy so that E(Nmin) = 0
    #energy2_master = energy2_master - 2.*energy_master[0]*energy_master + energy_master[0]**2
    energy = energy - energy[0]
            
    return N, lnPi, energy, beta, lnZ, volume
