from email.mime import application, base
from socket import send_fds
from sys import setrecursionlimit
from turtle import down
import numpy as np
from realSim import get_mSF, simulator_base,downsample
from trialSimulator import runSet, applyDrug,makeTrial_defaults,getPC
from trialSimulator import calculate_fisher_exact_p_value, calculate_MPC_p_value
from weargroup import make_multi_diaries
from joblib import Parallel, delayed
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
from numpy.random import default_rng
import scipy.stats as stats

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  


def get_a_patient(trialDur,clinTF,sensitivity,FAR):
    downsampleRATE=28
    howmanydays = downsampleRATE*trialDur
    e,c = make_multi_diaries(sampRATE=1,howmanydays=howmanydays,makeOBS=False,downsample_rate=downsampleRATE)
    # only pay attention to one of these
    if clinTF==True:
        X = c.copy()
    else:
        X = e.copy()
    # add sensitivity
    if sensitivity<1:
        Xs = applyDrug(efficacy=(1-sensitivity),x=X,baseline=0)
    else:
        Xs = X.copy()
    # add FAR
    if FAR>0:
        thisFAR = np.max([0,FAR + FAR*np.random.randn()])
        Xsf = Xs + downsampleRATE*thisFAR
    else:
        Xsf = Xs.copy()
    
    return Xsf
    
def get_a_qualified_patient(minSz,baseline,trialDur,clinTF,sensitivity,FAR):
    isDone = False
    counter = 0
    while isDone==False:
        this_pt = get_a_patient(trialDur,clinTF,sensitivity,FAR)
        isDone = np.mean(this_pt[0:baseline])>=minSz
        counter +=1
    return this_pt

def build_a_trial(N,halfN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR):
    trialDur = baseline+test
    
    
    # build trial data (number of patients by number of months
    trialData = np.zeros((N,trialDur))
    for counter in range(N):
        this_pt = get_a_qualified_patient(minSz,baseline,trialDur,clinTF,sensitivity,FAR)
        trialData[counter,:] = applyDrug(efficacy=PCB,x=this_pt,baseline=baseline)
        if counter>=halfN:
            trialData[counter,:] = applyDrug(efficacy=DRG,x=trialData[counter,:],baseline=baseline)
        
    PC = getPC(trialData,baseline,test)
    return PC

def didWeWin(PC,metricMPC_TF,halfN):
    if metricMPC_TF:
        # get MPC
        successTF = calculate_MPC_p_value(PC[:halfN],PC[halfN:]) < 0.05
    else:
        # get RR50
        successTF = calculate_fisher_exact_p_value(PC[:halfN],PC[halfN:]) < 0.05
        
    return successTF + 0

def findThresh(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,metricMPC_TF):
    
    allPCs = buildPCsets(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR)
    threshRR50 = checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF=False)
    threshMPC = checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF=True)
    
    return threshRR50,threshMPC

def buildPCsets(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR):
    N=maxN
    halfN=int(maxN/2)
    T1 = time.time()

    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(build_a_trial)(N,halfN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR) for _ in trange(REPS,desc='trials'))
    allPCs = np.array(temp,dtype=float)
    
    delT = np.round((time.time()-T1)/60)
    print(f'Calculating wins = {delT} minutes')

    print('Saving...',end='')
    np.save(fn,allPCs)
    print('.')
    return allPCs

def checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF):
    T1 = time.time()
    for thisN in trange(100,maxN,10,desc='findthresh'):
        halfN = int(thisN/2)
        wins = [ didWeWin(allPCs[iter,0:thisN],metricMPC_TF,halfN) for iter in range(REPS) ]
        thePow = np.mean(wins)
        if thePow>0.9:
            print('Thresh N = thisN')
            break
    delT = np.round((time.time()-T1)/60)
    print(f'Calculating wins = {delT} minutes')
    return thisN
