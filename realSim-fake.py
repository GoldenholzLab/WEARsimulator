# THIS IS A FAKE VERSION OF THE REALSIM.py 
# use this for testing only. NOT REAL!

import numpy as np


def simulator_base(sampRATE,number_of_days,cyclesTF=True,clustersTF=True, maxlimits=True, defaultSeizureFreq = -1,
    Lparams=[1,1,1,1,1,1],CP=[],returnDetails=False,clusterParams=[1,1,1,1,1],
    bestM=1.0,bestS = 1.0):

    if defaultSeizureFreq==-1:
        mSF = get_mSF(requested_msf=-1)
    SF = 30*mSF
    howmany = int(number_of_days*sampRATE)
    x = np.random.poisson(lam = SF, size=howmany)
    return x

def get_mSF(requested_msf,bestM=1,bestS=1):
    if requested_msf==-1:
        mSF = np.random.random()*9+1
    else:
        mSF = requested_msf
        
    return mSF


def downsample(x,byHowmuch):
    # input: 
    #    x = diary
    #    byHowMuch = integeter by how much to downsample
    # outputs
    #   x3 = the new diary, downsampled.
    #
    # If I sample 24 samples per day, and downsample by 24 then I get
    # daily samples as the output, for instance.
    #
    L = len(x)
    x2 = np.reshape(x,(int(L/byHowmuch),byHowmuch))
    x3 = np.sum(x2,axis=1)
    return x3