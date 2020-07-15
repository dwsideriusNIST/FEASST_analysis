import sys
import os
import numpy as np
import feasst as fst
import pyfeasst as pyfst
sys.path.insert(0, fst.install_dir() + '/plugin/monte_carlo/tutorial/')
import analyze
import copy

from ._append_operations import *

def data_abstraction(source,prefix,suffix):

    # Correct paths that do not end in /
    if source[-1] != '/': source+='/'

    # Determine restart file structure/pattern
    input_files = [ x for x in os.listdir(source) if (x.startswith(prefix) and x.endswith(suffix)) ]
    windows = len(input_files)

    # Stitch the windows together
    for window in range(windows):
        file_stub = source+prefix+str(window)+suffix
        #print(file_stub)
    
        monte_carlo = fst.MonteCarlo().deserialize(pyfst.read_checkpoint(file_stub))
        criteria = fst.FlatHistogram(monte_carlo.criteria())

        # Currently, this is only for fully-grown macrostates (expanded ensemble NOT okay)
        N_w = [ int(criteria.macrostate().value(state)) for state in range(criteria.num_states()) ]
        lnPi_w = [  criteria.bias().ln_prob().value(state) for state in range(criteria.num_states()) ]

        energy_analyzer = monte_carlo.analyze(monte_carlo.num_analyzers() - 1)
        energy_w = [ energy_analyzer.analyze(state).accumulator().average()
                     for state in range(criteria.num_states()) ]

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
        
    # Ensemble Constraints
    volume = monte_carlo.configuration().domain().volume()
    beta = monte_carlo.criteria().beta()
    lnZ = monte_carlo.criteria().beta_mu()

    # #Convert to NumPy Arrays
    N = np.array(N)
    lnPi = np.array(lnPi)
    energy = np.array(energy)
    
    # # Normalize lnPi
    lnPi= lnPi - max(lnPi)
    lnPi = lnPi - np.log(sum(np.exp(lnPi)))
    
    # Adjust Energy so that E(Nmin) = 0
    #energy2_master = energy2_master - 2.*energy_master[0]*energy_master + energy_master[0]**2
    energy = energy - energy[0]
    
    return N, lnPi, energy, beta, lnZ, volume
