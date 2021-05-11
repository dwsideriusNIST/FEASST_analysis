import sys
import os
import numpy as np
import pandas as pd
import feasst as fst
import pyfeasst as pyfst
#sys.path.insert(0, fst.install_dir() + '/plugin/monte_carlo/tutorial/')
#import analyze
import copy
from sys import platform
import re
import shutil

from ._append_operations import append_data

def fix_checkpoint(file_stub):
    with open(file_stub) as handle:
        checkpoint_string = handle.read()

    if not 'RandomMT19937' in checkpoint_string:
        return file_stub
    else:
        pass
        
    MT_serial = re.findall(r'(?<=RandomMT19937).*?(?=EndRandomMT19937)', checkpoint_string)
    #print(MT_serial)
    MT_serial_len = len(MT_serial[0].split())

    if platform == 'darwin':
        #length should be 631
        if MT_serial_len == 631:
            new_file = file_stub
        elif MT_serial_len == 632:
            # raise Exception('need to fix the checkpoint file')
            new_file = file_stub+'.fixed'
            # Fix linux-origin file
            print('fixing checkpoint: ', file_stub)
            with open(new_file,mode='w') as handle:
                handle.write(checkpoint_string.split('RandomMT19937')[0])
                handle.write('RandomMT19937 ')
                for i, word in enumerate(MT_serial[0].split()):
                    if i < 631:
                        handle.write(word+' ')
                handle.write('EndRandomMT19937')
                handle.write(checkpoint_string.split('EndRandomMT19937')[-1])
        else:
            raise Exception('ERROR: checkpoint has unknown origin')
    elif platform == 'linux':
        #length should be 632
        if MT_serial_len == 632:
            new_file = file_stub
        elif MT_serial_len == 631:
            # raise Exception('need to fix the checkpoint file')
            new_file = file_stub+'.fixed'
            # Fix mac-origin file
            print('fixing checkpoint: ', file_stub)
            with open(new_file,mode='w') as handle:
                handle.write(checkpoint_string.split('RandomMT19937')[0])
                handle.write('RandomMT19937 ')
                for word in enumerate(MT_serial[0].split()):
                    handle.write(word+' ')
                handle.write('574 ')
                handle.write('EndRandomMT19937')
                handle.write(checkpoint_string.split('EndRandomMT19937')[-1])
        else:
            raise Exception('ERROR: checkpoint has unknown origin')   
    else:
        raise Exception('ERROR: unknown platform', platform)

    return new_file
    

def data_abstraction(source,prefix,suffix,splice_type='smoothed'):

    # Correct paths that do not end in /
    if source[-1] != '/': source+='/'

    # Determine restart file structure/pattern
    input_files = [ x for x in os.listdir(source) if (x.startswith(prefix) and x.endswith(suffix)) ]
    windows = len(input_files)

    # Stitch the windows together
    for window in range(windows):
        file_stub = source+prefix+str(window)+suffix
        #print(file_stub)

        #--------------------------------------------
        # Check Mersenne Twister version
        file_to_read = fix_checkpoint(file_stub)        
        monte_carlo = fst.MonteCarlo().deserialize(pyfst.read_checkpoint(file_to_read))
        if file_to_read != file_stub:
            os.remove(file_to_read)
        criteria = fst.FlatHistogram(monte_carlo.criteria())
            
        # Currently, this is only for fully-grown macrostates (expanded ensemble NOT okay)
        N_w = [ int(criteria.macrostate().value(state)) for state in range(criteria.num_states()) ]
        lnPi_w = [  criteria.bias().ln_prob().value(state) for state in range(criteria.num_states()) ]

        energy_analyzer = monte_carlo.analyze(monte_carlo.num_analyzers() - 1)
        energy_w = [ energy_analyzer.analyze(state).accumulator().average()
                     for state in range(criteria.num_states()) ]
        energy2_w = [ energy_analyzer.analyze(state).accumulator().moment(1)
                      /energy_analyzer.analyze(state).accumulator().num_values()
                     for state in range(criteria.num_states()) ]
        
        if window == 0:
            # Deep copy first window into master arrays
            N = copy.deepcopy(N_w)
            lnPi = copy.deepcopy(lnPi_w)
            energy = copy.deepcopy(energy_w)
            energy2 = copy.deepcopy(energy2_w)
        else:
            # Append to master arrays
            Nold = [x for x in N] #storage
            N, lnPi = append_data(N,lnPi,N_w,lnPi_w,splice_type='smoothed')
            Ntmp, energy = append_data(Nold,energy,N_w,energy_w,splice_type=splice_type)
            Ntmp, energy2 = append_data(Nold,energy2,N_w,energy2_w,splice_type=splice_type)

    # Ensemble Constraints
    volume = monte_carlo.configuration().domain().volume()
    beta = monte_carlo.thermo_params().beta()
    lnZ = monte_carlo.thermo_params().beta_mu()

    # #Convert to NumPy Arrays
    N = np.array(N)
    lnPi = np.array(lnPi)
    energy = np.array(energy)
    energy2 = np.array(energy2)
    
    # # Normalize lnPi
    lnPi= lnPi - max(lnPi)
    lnPi = lnPi - np.log(sum(np.exp(lnPi)))
    
    # Adjust Energy so that E(Nmin) = 0
    #  NOTE: This does not affect the moments
    energy2 = energy2 - 2.*energy[0]*energy + energy[0]**2
    energy = energy - energy[0]
    
    return N, lnPi, energy, energy2, beta, lnZ, volume



def data_abstraction_logs(source,prefix,suffix,splice_type='smoothed'):

    # Correct paths that do not end in /
    if source[-1] != '/': source+='/'

    # Determine log structure/pattern
    input_files = [ x for x in os.listdir(source) if (x.startswith(prefix) and x.endswith('_crit'+suffix)) ]
    windows = len(input_files)
    #print(windows)

    # Stitch the windows together
    for window in range(windows):
        # Read the criteria file
        crit_file = source + prefix + str(window) + '_crit'+suffix
        #print(crit_file)
        skip = [0] # we have to skip some number of lines, but how many depends on the bias mode
        done = False
        while not done:
            with open(crit_file,mode='r') as handle:
                criteria_data = pd.read_csv(handle, skiprows=skip)
            try:
                N_w = np.array(criteria_data['state'])
                done = True
            except:
                skip = skip + [len(skip)]
        #display(criteria_data)
        if 'ln_prob.1' in criteria_data.columns:
            # this covers WLTM cases
            lnPi_w = np.array(criteria_data['ln_prob.1'])
        else:
            lnPi_w = np.array(criteria_data['ln_prob'])

        # Read the energy file
        #  This starts with the assumption that the sim was in "phase1"
        energy_file = source + prefix + str(window) + '_energy'+suffix+'_phase1'
        if not os.path.exists(energy_file):
            # fall back to non-phase energy file
            energy_file = source + prefix + str(window) + '_energy'+suffix
            #print(energy_file)
        with open(energy_file,mode='r') as handle:
            energy_data = pd.read_csv(handle)
        #display(energy_data)

        energy_w = [ energy_data['moment0'][x]/float(energy_data['n'][x]) for x in energy_data['state'] ]
        energy2_w = [ energy_data['moment1'][x]/float(energy_data['n'][x]) for x in energy_data['state'] ]

        # Append data to the master arrays
        if window == 0:
            # Deep copy first window into master arrays
            N = copy.deepcopy(N_w)
            lnPi = copy.deepcopy(lnPi_w)
            energy = copy.deepcopy(energy_w)
            energy2 = copy.deepcopy(energy2_w)
        else:
            # Append to master arrays
            Nold = [x for x in N] #storage
            N, lnPi = append_data(N,lnPi,N_w,lnPi_w,splice_type='smoothed')
            Ntmp, energy = append_data(Nold,energy,N_w,energy_w,splice_type=splice_type)
            Ntmp, energy2 = append_data(Nold,energy2,N_w,energy2_w,splice_type=splice_type)

    # #Convert to NumPy Arrays
    N = np.array(N)
    lnPi = np.array(lnPi)
    energy = np.array(energy)
    energy2 = np.array(energy2)

    # # Normalize lnPi
    lnPi= lnPi - max(lnPi)
    lnPi = lnPi - np.log(sum(np.exp(lnPi)))

    # Adjust Energy so that E(Nmin) = 0
    #  NOTE: This does not affect the moments
    energy2 = energy2 - 2.*energy[0]*energy + energy[0]**2
    energy = energy - energy[0]

    return N, lnPi, energy, energy2
