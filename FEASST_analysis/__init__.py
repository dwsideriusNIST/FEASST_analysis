import feasst
import lnPi
import os
import copy
import numpy as np

def append_data(N0,array0,N1,array1,smooth_type="mean"):
    #Splices array1 onto array0, with smoothing
    N_out = [x for x in N0]
    array_out = [x for x in array0]
    ref_min = N1[0]
    ref_max = N0[-1]
    for Ni in N1[1:]:  #start with the second position
        if Ni <= ref_max:
            if smooth_type == "mean":
                delta = 0.5*( (array0[Ni]-array0[Ni-1]) + (array1[Ni-ref_min]-array1[Ni-ref_min-1])  )
            else:
                raise Exception("Unknown smoothing type: "+smooth_type)
            array_out[Ni] = array_out[Ni-1]+delta
        else:
            delta = (array1[Ni-ref_min]-array1[Ni-ref_min-1])
            array_out.append(array_out[-1]+delta)
            N_out.append(N_out[-1]+1)
    return N_out, array_out

def append_energy(N0,array0,N1,array1):
    #Splices array1 onto array0, with smoothing
    N_out = [x for x in N0]
    array_out = [x for x in array0]
    ref_min = N1[0]
    ref_max = N0[-1]
    for Ni in N1[1:]:  #start with the second position
        if Ni > ref_max:
            array_out.append(array1[Ni-ref_min])
            N_out.append(N_out[-1]+1)
    return N_out, array_out

class feasst_analysis(lnPi.lnPi_phases):
    def __init__(self,
                 base,
                 phases='get',
                 argmax='get',
                 argmax_kwargs=None,
                 phases_kwargs=None,
                 build_kwargs=None,
                 ftag_phases=None,
                 ftag_phases_kwargs=None):
        super(feasst_analysis, self).__init__(base,
                                              phases=phases,
                                              argmax=argmax,
                                              argmax_kwargs=argmax_kwargs,
                                              phases_kwargs=phases_kwargs,
                                              build_kwargs=build_kwargs,
                                              ftag_phases=ftag_phases,
                                              ftag_phases_kwargs=ftag_phases_kwargs)
        self.energy = np.array([])
        self.energy2 = np.array([])
        self.Sx = np.array([])
        self.canSx = False
        self.extrapolated = False
        
    @classmethod
    def from_restart(cls,
                     source,
                     prefix,
                     ftag_phases=None):
        #Constructor method to build object from FEASST restart file(s)
        
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
        
        # Convert lnPi to format required by lnPi Class
        lnPi_data = np.array([ [float(Ni), lnPii] for Ni,lnPii in zip(N,lnPi) ])
        
        #Build class object using "from_data" method in parent class
        child = cls.from_data(lnPi_data,
                              mu=lnZ/beta,
                              volume=volume,
                              beta=beta,
                              num_phases_max=2,
                              argmax_kwargs=dict(min_distance=[5,10,20,40]),
                              ftag_phases=ftag_phases)
        child.energy = np.array(energy)
        
        return child

    # Override the base class reweight method because we need the returned object to carry sublcass attributes
    #  NOTE: If it is necessary to derive a class *from this derived class*, then a new reweight method
    #        will be needed
    def reweight(self,
                 muRW,
                 ZeroMax=True,
                 CheckTail=False,
                 Threshold=10.,
                 Pad=False,
                 **kwargs):
        # Override the base class method because we need the returned object to carry subclass attributes
        child = self.copy(
            base=self.base.reweight(muRW, ZeroMax=ZeroMax, Pad=Pad, **kwargs),
            phases='get',
            argmax='get')
        # Copy new_class attributes to child object
        child.energy = copy.deepcopy(self.energy)
        child.energy2 = copy.deepcopy(self.energy2)
        child.Sx = copy.deepcopy(self.Sx)
        child.canSx = copy.deepcopy(self.canSx)

        # Confirm that the tail of lnPi satisfies threshold criteria
        if CheckTail:
            maxima = child.base._peak_local_max()[0][0] #what about 2-D?
            delta = np.abs(child.base.data[-1] - child.base.data[max(maxima)])
            if delta < Threshold:
                raise ValueError('Insufficient tail in lnPi: '+str(delta)+'\n Reweighted to: '+str(muRW))
        
        return child
    
#    def extrapolate(self,
#                    beta_extrap,
#                    Normalize=False,
#                    CheckTail=False,
#                    CheckExtrap=True):
#    
#        if self.extrapolated and CheckExtrap:
#            raise AttributeError('Extrapolating already *extrapolated* lnPi is dangerous. Disable via the CheckExtrap=False flag')
#    
#        try:
#            if self.energy2.shape != self.base.data.shape:
#                raise AttributeError('Canonical U^2 array is wrong size')
#        except:
#            raise AttributeError('Extrapolation requires input of <U^2>')
#        
#        child.canSx = False # Force recalculation of the canonical entropy
#        child.Sx = np.array([]) ## HOW DO WE SHAPE THIS?
#        child.extrapolated = True
#    
#        return child
    
    # PROPERTIES
    #FIGURE OUT HOW TO USE WPK'S CACHE OPS TO SAVE TIME HERE

    def generic_property(self,canonical_property):
        try:
            if canonical_property.shape != self.base.data.shape:
                raise AttributeError('Canonical property array is wrong size')
            if type(canonical_property) != np.ndarray:
                raise AttributeError('Canonical property array must be a NumPy array')
        except:
            raise AttributeError('Unknown canonical property')
        Prop_avg = np.array([ (x.pi_norm*canonical_property).sum(axis=-1) for x in self ])
        return Prop_avg
            
    @property
    def Uaves(self):
        return self.generic_property(self.energy)

    @property
    def Saves_Gibbs(self):
        # Gibbs Entropy
        if not self.canSx:
            # Compute the Canonical Entropies
            self.canonical_Sx
        S_avg = self.generic_property(self.Sx) 
        return S_avg

    @property
    def Saves_Boltzmann(self):
        #Boltzmann Entropy
        S_avg = self.base.beta*(self.Uaves - np.dot(self.base.mu,self.Naves) - self.Omegas())
        return S_avg
    
    @property
    def canonical_Sx(self):
        try:
            if self.energy.shape != self.base.data.shape:
                raise AttributeError('Canonical energy array is wrong size')
        except:
            raise AttributeError('User must input canonical energy before requesting canonical entropy')
        Sx = self.base.data - self.base.data.ravel()[0] + self.base.beta*self.energy \
             - self.base.beta*np.dot(self.base.mu,self.base.coords)       
        self.Sx = Sx
        self.canSx = True
        return Sx    
