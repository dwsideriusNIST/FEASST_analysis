def append_data(N0,array0,N1,array1,splice_type='smoothed'):
    if splice_type == 'smoothed':
        return append_data_smoothed(N0,array0,N1,array1)
    elif splice_type == 'smoothed2':
        return append_data_smoothed2(N0,array0,N1,array1)
    elif splice_type == 'mean':
        return append_data_mean(N0,array0,N1,array1)
    elif splice_type == 'pieced':
        return append_data_pieced(N0,array0,N1,array1)
    else:
        raise Exception("Unknown splicing type: "+splice_type)

def append_data_pieced(N0,array0,N1,array1):
    # Splices array1 onto array0
    #  No smoothing - just directly splice the 'next' window

    # Create output lists
    N_out = [x for x in N0]
    array_out = [x for x in array0]
    
    # Determine the amount of overlap
    N_overlap = sorted([x for x in set(N0).intersection(N1)])
    
    ref_min = N1[0]

    for Ni in N1[1:]:  #start with the second position in the new array
        # determine a delta function, based on weighting between the overlapping windows
        if Ni <= N_overlap[-1]:
            pass
        else:
            array_out.append( array1[Ni-ref_min] )
            N_out.append(N_out[-1]+1)
    return N_out, array_out

def append_data_mean(N0,array0,N1,array1):
    # Splices array1 onto array0
    #  overlap points are averaged together

    # Create output lists
    N_out = [x for x in N0]
    array_out = [x for x in array0]
    
    # Determine the amount of overlap
    N_overlap = sorted([x for x in set(N0).intersection(N1)])
    
    ref_min = N1[0]

    for Ni in N1[1:]:  #start with the second position in the new array
        # determine a delta function, based on weighting between the overlapping windows
        if Ni <= N_overlap[-1]:
            weight = 0.50
            delta = (1. - weight)*(array0[Ni] - array0[Ni-1]) + weight*(array1[Ni-ref_min] - array1[Ni-ref_min-1])
            array_out[Ni] = array_out[Ni-1] + delta
        else:
            array_out.append( array_out[-1] + array1[Ni-ref_min] - array1[Ni-ref_min-1] )
            N_out.append(N_out[-1]+1)
    return N_out, array_out

def append_data_smoothed(N0,array0,N1,array1):
    # Splices array1 onto array0
    #  Smoothing is determined by amount of overlap

    # Create output lists
    N_out = [x for x in N0]
    array_out = [x for x in array0]
    
    # Determine the amount of overlap
    N_overlap = sorted([x for x in set(N0).intersection(N1)])
    
    ref_min = N1[0]

    for Ni in N1[1:]:  #start with the second position in the new array
        # determine a delta function, based on weighting between the overlapping windows
        if Ni <= N_overlap[-1]:
            weight = float(Ni - N_overlap[1]) / float(len(N_overlap)-2)
            delta = (1. - weight)*(array0[Ni] - array0[Ni-1]) + weight*(array1[Ni-ref_min] - array1[Ni-ref_min-1])
            array_out[Ni] = array_out[Ni-1] + delta
        else:
            array_out.append( array_out[-1] + array1[Ni-ref_min] - array1[Ni-ref_min-1] )
            N_out.append(N_out[-1]+1)

    return N_out, array_out


def append_data_smoothed2(N0,array0,N1,array1):
    # Splices array1 onto array0
    #  Weighted average of the overlap regions

    # Create output lists
    N_out = [x for x in N0]
    array_out = [x for x in array0]
    
    # Determine the amount of overlap
    N_overlap = sorted([x for x in set(N0).intersection(N1)])
    
    ref_min = N1[0]

    for Ni in N1[1:]:  #start with the second position in the new array
        # determine a delta function, based on weighting between the overlapping windows
        if Ni <= N_overlap[-1]:
            weight = float(Ni - N_overlap[1]) / float(len(N_overlap)-2)
            array_out[Ni] = (1. - weight)*array_out[Ni] + weight*array1[Ni-ref_min]
        else:
            array_out.append( array1[Ni-ref_min] )
            N_out.append(N_out[-1]+1)
    return N_out, array_out
