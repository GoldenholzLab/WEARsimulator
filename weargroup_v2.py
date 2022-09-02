from email.mime import application, base
from socket import send_fds
from sys import setrecursionlimit
from turtle import down
import numpy as np
from realSim import get_mSF, simulator_base,downsample
from trialSimulator import runSet,makeTrial_defaults,getPC
from trialSimulator import calculate_fisher_exact_p_value, calculate_MPC_p_value
from weargroup import make_multi_diaries
from joblib import Parallel, delayed
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import seaborn as sns
import pandas as pd
from numpy.random import default_rng
import scipy.stats as stats

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  


def applyDrug(efficacy,x,baseline):
    # INPUTS:
    #  efficacy = fraction of seziures removed
    #  x = diary
    #  baseline = number of samples to consider as baseline samples
    #     that do not get drug applied at all

    if efficacy>0:    
        # ie don't do anything if efficacy is zero
        x2 = x.copy()
        L = len(x)

        # for each seizure count entry, and for each seizure, consider deleting
        # using efficacy as the probability of deleting
        for iter in range(baseline,L):
            x2[iter] -= np.sum(np.random.random(int(x[iter]))<efficacy)

        return x2
    else:
        # DO NOITHING for efficacy = 0
        return x

def get_a_patient(trialDur,baseline,clinTF,sensitivity,FAR,PCB,DRG,useDrug,inflater):
    downsampleRATE=28
    howmanydays = downsampleRATE*trialDur
    e,c = make_multi_diaries(sampRATE=1,howmanydays=howmanydays,makeOBS=False,downsample_rate=downsampleRATE)
    # only pay attention to one of these
    if clinTF==True:
        X = c.copy()
    else:
        X = e.copy()
        
    # FIRST, one must add drug and placebo.
    # The drug is internal. The detector device is after those internal chanegs were made.
    X = applyDrug(efficacy=PCB,x=X,baseline=baseline)
    if useDrug:
        X = applyDrug(efficacy=DRG,x=X,baseline=baseline)
        
    # add sensitivity
    if sensitivity<1:
        Xs = applyDrug(efficacy=(1-sensitivity),x=X,baseline=0)
    else:
        Xs = X.copy()
    # add FAR
    if FAR>0:
        L = len(X)
        downsampleRATEhalf = 0.5*downsampleRATE
        Xadder = np.zeros(L)
        for i in range(L):
            x = np.random.random(downsampleRATE)
            zeroMean = FAR*inflater*(np.sum(x) - downsampleRATEhalf)
            Xadder[i] = np.round(downsampleRATE*FAR + zeroMean)
            if Xadder[i]<0:
                Xadder[i] = 0
        Xs += Xadder
    return Xs


def get_a_qualified_patient(minSz,baseline,trialDur,clinTF,sensitivity,FAR,PCB,DRG,useDRG,inflater):
    isDone = False
    while isDone==False:
        this_pt = get_a_patient(trialDur,baseline,clinTF,sensitivity,FAR,PCB,DRG,useDRG,inflater)
        isDone = np.mean(this_pt[0:baseline])>=minSz

    return this_pt

def build_a_trial(N,halfN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater):
    trialDur = baseline+test
    
    # build trial data (number of patients by number of months
    trialData = np.zeros((N,trialDur))
    for counter in range(N):
        trialData[counter,:] = get_a_qualified_patient(minSz,baseline,trialDur,clinTF,sensitivity,FAR,PCB,DRG,(counter>=halfN),inflater)
                
    PC = getPC(trialData,baseline,test)
    return PC

def didWeWin(PC_a,PC_b,metricMPC_TF):
    if metricMPC_TF:
        # get MPC
        successTF = calculate_MPC_p_value(PC_a,PC_b) < 0.05
    else:
        # get RR50
        successTF = calculate_fisher_exact_p_value(PC_a,PC_b) < 0.05
        
    return successTF + 0

def findThresh(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,recalc=True,inflater=2):
    
    if recalc==True:
        allPCs = buildPCsets(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater)
    else:
        print('Loading...',end='')
        allPCs = np.load(fn) 
        print('done.')
    threshRR50 = checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF=False)
    threshMPC = checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF=True)
    
    return threshRR50,threshMPC

def buildPCsets(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater):
    N=maxN
    halfN=int(maxN/2)
    T1 = time.time()

    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(build_a_trial)(N,halfN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater) for _ in trange(REPS,desc='trials'))
    allPCs = np.array(temp,dtype=float)
    
    delT = np.round((time.time()-T1)/60)
    print(f'Calculating wins = {delT} minutes')

    print('Saving...',end='')
    np.save(fn,allPCs)
    print('.')
    return allPCs

def checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF):
    T1 = time.time()
    print('Threshold ...',end='')
    halfMax = int(maxN/2)
    for thisN in range(100,maxN,5):
        halfN = int(thisN/2)
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(didWeWin)(allPCs[iter,0:halfN],allPCs[iter,halfMax:(halfMax+halfN)],metricMPC_TF) for iter in range(REPS))
        wins = np.array(temp,dtype=int)
    
        thePow = np.mean(wins)
        if thePow>0.9:
            break
    delT = np.round((time.time()-T1)/60)
    print(f'{thisN}. runtime = {delT} minutes')
    return thisN

def buildSET_of_N(senseLIST,farLIST,recalc,thiscsv,clinTF=True,REPS=10000,maxN=3000,DRG=0.2,mini=False,inflater=2):
    if mini==False:
        numCPUs = 9
        thedir = '/Users/danielgoldenholz/Library/CloudStorage/OneDrive-BethIsraelLaheyHealth/Comprehensive Epilepsy Program & EEG Lab/Research/Goldenholz Lab/wear'
    else:
        numCPUs = 7
        thedir = '/Users/dgodenh/OneDrive - Beth Israel Lahey Health/Comprehensive Epilepsy Program & EEG Lab/Research/Goldenholz Lab/wear'

    d2 = pd.DataFrame()
    for sensitivity in senseLIST:
        for FAR in farLIST:
            fn = f'{thedir}/PC_{clinTF}_sens{sensitivity}_FAR{FAR}_{REPS}x{maxN}_{inflater}.npy'
            tRR,tMP = findThresh(fn=fn,numCPUs=numCPUs,REPS=REPS,maxN=maxN,DRG=DRG,PCB=0,minSz=4,
                    baseline=2,test=3,clinTF=clinTF,sensitivity=sensitivity,FAR=FAR,recalc=recalc,inflater=inflater)
            #print(f'S={sensitivity} F={FAR} Threshold RR50 = {tRR} Threshold MPC = {tMP}')
            df = pd.DataFrame({'sensitivity':[sensitivity],
                               'FAR':[FAR],
                               'RR50':[tRR],
                               'MPC':[tMP]})
            print(df)
            d2 = pd.concat([d2,df])
    d2.to_csv(thiscsv,index_label=False)
    print(d2)
    
def drawGrid(fn,clinTF,ax=[]):
    d2=pd.read_csv(fn)
    if ax==[]:
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))     
        doShow = True
    else:
        doShow = False
    #maxList = [530, 900]
    maxList = [550,910]
    mlist=['MPC','RR50']
    for mi,metric_type in enumerate(mlist):
        d3 = d2.copy()
        d3[metric_type] = np.round(100*d2[metric_type] / maxList[mi]).astype('int')
        d3=d3.drop(columns=mlist[1-mi])
        thispow = d3.pivot('FAR','sensitivity',metric_type)
        sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=ax[mi],vmin=0,vmax=100)
        ax[mi].set_title(f'Power Metric={metric_type} clinTF={clinTF}')    
    if doShow==True:
        plt.show()
    