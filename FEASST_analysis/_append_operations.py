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
