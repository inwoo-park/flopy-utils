#!/usr/bin/env python
import numpy as np
import pandas
import matplotlib.pyplot as plt

def find_bc(mf,layer=0,debug=0):# {{{
    '''
    Explain
     get boundary of modflow grid.
    Usage
     ibc = find_bc(mf)
    '''
    # check whether input model is modflow2005-family or modflow6?
    # get boundary information
    if isinstance(mf,flopy.modflow.Modflow):
        ibound = mf.bas6.ibound.array[layer,:,:]
    elif isinstance(mf,flopy.mf6.ModflowGwf):
        ibound = mf.dis.idomain.array[layer,:,:]
    else:
        raise Exception('current input "mf" class is {}'.format(type(mf)))
    # where is boundary information?
    if debug:
        print(ibound)
    nrow, ncol = np.shape(ibound)
    ibc = np.zeros((nrow,ncol))
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if ibound[i,j] > 0:
                if (ibound[i+1,j]==0) or (ibound[i,j+1]==0) or (ibound[i-1,j]==0) or (ibound[i,j-1]==0):
                    ibc[i,j]=1
    return ibc
