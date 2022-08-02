import enum
import numpy as np
from realSim import get_mSF, simulator_base,downsample
from trialSimulator import runSet, applyDrug,makeTrial_defaults,getPC
from trialSimulator import calculate_fisher_exact_p_value, calculate_MPC_p_value
from joblib import Parallel, delayed
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd



np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  


def add_FAR(x,FAR):
    # INPUTS:
    #   x - the diary
    #   FAR - the rate of false alarms per sample
    # OUTPUTS:
    #   newx - the diary with extra alarms in there
    
    added_sz = np.round(np.random.rand(len(x))*FAR)
    
    return x + added_sz

def make_multi_diaries(sampRATE,howmanydays,makeOBS):
    # INPUTS:
    #  sampRATE = samples per day
    #  howmanydays = how many days to generate
    #  makeOBS = True: make observed_dairy, False: don't    
    #
    # OUTPUTS:
    #  true_e_diary - true electrographic seizures (including clinical)
    #  true_clin_diary - true clinical only seizures
    #  observed_clin_diary - observed clinical only seizures
    #
    #
    # USAGE:
    #true_e_diary, true_clin_diary, observed_clin_diary =  make_multi_diaries(sampRATE,howmanydays)
    
    # CONSTANTS
    esz_vs_all_sz_mean = 0.39   # Neurovista result
    esz_vs_all_sz_std = 0.22    # Neurovista result
    obs_sensitivity = 0.5   # Elgar 2018
    obs_FAR = 0.0           # ??
    
    # generate a monthly seizure frequency that is realistic
    mSF = get_mSF( requested_msf=-1 )
    
    # increase true SF to account for under-reporting
    mSF /= obs_sensitivity
    
    # decrease true SF to account for over-reporting
    mSF *= (1 + obs_FAR) 

    # account for difference intracranial vs clinical
    esz_vs_all_sz = esz_vs_all_sz_std * np.random.randn() + esz_vs_all_sz_mean
    esz_vs_all_sz = np.min([np.max([.03,esz_vs_all_sz]),.72])       # error check
    #  NV data has limits 0.3 - 0.72
    
    
    # eFactor = multiply it by # clin szs and give # of e-seizres+clini szs
    eFactor =  (1 / (1 - esz_vs_all_sz) )    
    mSF_all = mSF * eFactor
    
    # generate true electrographic diary (which includes true clin szs too)
    true_e_diary = simulator_base(sampRATE=sampRATE,number_of_days=howmanydays,defaultSeizureFreq=mSF_all)
    
    # remove a percentage of the complete set of seizures to get the true clinical set
    efficacy = esz_vs_all_sz
    true_clin_diary = applyDrug(efficacy=efficacy,x=true_e_diary,baseline=0)
    
    if makeOBS:
        # calculate the observed clinical seizures
        observed_clin_diary = remove_and_add(true_clin_diary,obs_sensitivity,obs_FAR,sampRATE)
        
        return true_e_diary, true_clin_diary, observed_clin_diary
    else:
        return true_e_diary, true_clin_diary


def remove_and_add(diary,sensitivity,dailyFAR,sampRATE):
        # INPUT
    #  diary - input diary
    #  sensivity - how sensitive is this detector?
    #  dailyFAR - the false alarm rate, per day
    #  sampRATE - number of samples per day
    # OUTPUT
    #  newdiary
    
    temp = applyDrug(efficacy=1-sensitivity,x=diary,baseline=0)
    newdiary = add_FAR(x = temp,FAR = dailyFAR/sampRATE)
    return newdiary

def build_full_set_of_diaries(sampRATE,howmanydays,clin_sensitivity,clin_FAR,e_sensitivity,e_FAR):
    # INPUTS:
    #  sampRATE = samples per day
    #  howmanydays = how many days to generate
    #  clin_sensitivity = fraction of clinical seizures detected by this detector 
    #  clin_FAR = false alarm rate for clinical seizures (rate per DAY)
    #  e_sensitivity = fraction of the electrographic seizures detected by this detector
    #  e_FAR = false alarm rate for electrographic seizures (rate per DAY)
    # OUTPUTS:
    #  true_e_diary - true electrographic seizures (including clinical)
    #  true_clin_diary - true clinical only seizures
    #  observed_clin_diary - observed clinical seizures
    #  detector_e_diary - diary from detector
    #  
    
    true_e_diary, true_clin_diary,observed_clin_diary =  make_multi_diaries(sampRATE,howmanydays,makeOBS=True)
    
    detector_e_diary = remove_and_add(true_e_diary,e_sensitivity,e_FAR,sampRATE)
    detector_clin_diary = remove_and_add(true_clin_diary,clin_sensitivity,clin_FAR,sampRATE)

    return true_e_diary, true_clin_diary, observed_clin_diary, detector_e_diary, detector_clin_diary

def build_a_kind_of_diary(sampRATE,howmanydays,sensitivity,FAR,clinTF):
    # INPUTS:
    #  sampRATE = samples per day
    #  howmanydays = how many days to generate
    #  sensitivity = fraction of clinical seizures detected by this detector 
    #  FAR = false alarm rate for clinical seizures (rate per DAY)
    #  clinTF = True: clin, False: e-szs
    # OUTPUTS:
    #  diary - the requested kind
    
    true_e_diary, true_clin_diary =  make_multi_diaries(sampRATE,howmanydays,makeOBS=False)
    if clinTF==False:
        diary = remove_and_add(true_e_diary,sensitivity,FAR,sampRATE)
    else:
        diary = remove_and_add(true_clin_diary,sensitivity,FAR,sampRATE)
    return diary
    


def get_pow_kind_full(N,numCPUs,REPS,DRG,sensitivity,FAR,clinTF,metric_type):
    # INPUTS:
    # N - total number of patients
    # metric_type - 'MPC' or 'RR50'
    # numCPUs 
    # REPS - how many trials to run
    # DRG fraction of time drug works
    # sensitivity - how sensitive is device
    # FAR - false alarm rate
    # clinTF - clinical or e-szs
    # metricType - 'RR50' or 'MPC'
    #
    # OUTPUTS:
    #  pow = the % of trials that successfully distinguish drug from placebo
    repSET = 400
    minCHANGE = 0.03
    subREPS = int(REPS/repSET)
    finalpow = 0
    allDone=0
    soFar = 0
    for K in range(subREPS):
        pow = get_pow_kind_sub(N,numCPUs,repSET,DRG,sensitivity,FAR,clinTF,metric_type)
        finalpow = (finalpow*soFar) + (pow*repSET)
        soFar += repSET
        finalpow /= soFar
        #print(f'K={K} finalpow={finalpow} pow={pow}')
        if np.abs(finalpow-pow)<minCHANGE:
            allDone +=1
        else:
            allDone=0     
        if allDone==3:
            break
    #print(f'Pow = {finalpow:0.2} totalREPS={soFar}. ',end='')
    return finalpow

def get_pow_kind_sub(N,numCPUs,REPS,DRG,sensitivity,FAR,clinTF,metric_type):
    # INPUTS:
    # N - total number of patients
    # metric_type - 'MPC' or 'RR50'
    # numCPUs 
    # REPS - how many trials to run
    # DRG fraction of time drug works
    # sensitivity - how sensitive is device
    # FAR - false alarm rate
    # clinTF - clinical or e-szs
    # metricType - 'RR50' or 'MPC'
    #
    # OUTPUTS:
    #  pow = the % of trials that successfully distinguish drug from placebo
    #  plist = array of TF, T=success
    
    minSz = 4
    PCB = 0.0
    baseline = 56
    test = 84    
    if numCPUs == 1:
        plist = [run1trial_kind(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF,metric_type) for _ in range(REPS)]
    else:
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            plist = par(delayed(run1trial_kind)(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF,metric_type) for _ in range(REPS))
    plist = np.array(plist,dtype=object)
    #plist = np.asarray(plist,dtype=object)
    pow = np.mean(plist<0.05,axis=0)
    return pow

def get_one_pow(N,minSz,DRG,PCB,baseline,test):
    
    trialData = makeTrial_defaults(minSz,N,DRG,PCB,baseline,test)
    PC = getPC(trialData,baseline,test)
    pRR50 = calculate_fisher_exact_p_value(PC[0:int(N/2)],PC[int(N/2):])
    pMPC = calculate_MPC_p_value(PC[0:int(N/2)],PC[int(N/2):])
    return [pRR50,pMPC]


                
def makeOnePt_kind(minSz,dur,baseline,sensitivity,FAR,clinTF):
    sampFREQ = 24
    notDone = True
    while (notDone==True):
        x = build_a_kind_of_diary(sampRATE=sampFREQ,howmanydays=dur,sensitivity=sensitivity,
            FAR = FAR, clinTF = clinTF)
        x2 = downsample(x,sampFREQ)
        if sum(x2[0:baseline])>=minSz:
            notDone = False
    return x2

def makeTrial_kind(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF):
    dur = baseline+test

    trialData = np.zeros((N,dur))
    for pt in range(N):
        temp = makeOnePt_kind(minSz,dur,baseline,sensitivity,FAR,clinTF)
        temp = applyDrug(PCB,temp,baseline)
        if pt>=(N/2):
            temp = applyDrug(DRG,temp,baseline)
        trialData[pt,:] = temp
    
    return trialData

def run1trial_kind(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF,metric_type):
    trialData = makeTrial_kind(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF)
    PC = getPC(trialData,baseline,test)
    if metric_type=='RR50':
        p = calculate_fisher_exact_p_value(PC[0:int(N/2)],PC[int(N/2):])
    else:
        p = calculate_MPC_p_value(PC[0:int(N/2)],PC[int(N/2):])
    return p

def run1Power_kind(REPS,numCPUs,minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF,metric_type):
    
    if numCPUs==1:
        X = [ run1trial_kind(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF,metric_type) for _ in range(REPS)] 
    else:
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                X = par(delayed(run1trial_kind)(minSz,N,DRG,PCB,baseline,test,sensitivity,FAR,clinTF,metric_type) for _ in range(REPS))
    plist = np.array(X,dtype=float)
    power = np.mean(plist<0.05)
    return power

def do_full_case1_sim(figname,numCPUs=9,REPS=5000,DRG=0.2): 
    T1 = time.time()
    highestN = 1000
    sensSET = [.33,.66,1]
    FARset = [0,.2,.4]
  
    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,10))
    
    metriclist= ['RR50','MPC']
    for im,metric_type in enumerate(tqdm(metriclist,desc='metric')):
        if im==1:
            Nlist = [75,100,125,150,175,200]
        else:
            Nlist=  [100,200,300,400,500,600,700,800]
            
        for ic,clinTF in enumerate(tqdm([True,False],desc='clinTF')):
        #ic = 0
            L = len(FARset)*len(sensSET)
            k = pd.DataFrame(np.zeros((L,3)),columns=['FAR','sensitivity','N'])
            ind = 0
            for fi,FAR in enumerate(tqdm(FARset,desc='FAR')):
                for si,sensitivity in enumerate(tqdm(sensSET,desc='sensitivity')): 
                    finalN = highestN
                    for N in Nlist:
                        p = get_pow_kind_full(N,numCPUs,REPS,DRG,sensitivity,FAR,clinTF,metric_type)
                        if p>0.9:
                            finalN=N
                            break
                    #print(f'\nFAR={FAR} sensitivity={sensitivity} FinalN={finalN}')
                    k.iloc[ind,:] = [FAR,sensitivity,finalN]
                    #thexy[ind,:] = [FAR,sensitivity,finalN]
                    ind+=1
            k['N'] = k['N'].astype('int')
            #k = pd.DataFrame({'FAR':thexy[0,:],'sensitivity':thexy[1,:],'N':thexy[2,:].astype('int')})
            thispow = k.pivot('FAR','sensitivity','N')
            print(k)
            sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=ax[im,ic])
            ax[im,ic].set_title(f'Power Metric={metric_type} clinTF={clinTF}')    
    plt.savefig(figname,dpi=300)
    plt.show()    
    print(f'Time = {time.time()-T1}')
    

def do_simple_case1_sim(numCPUs=9,REPS=5000,DRG=0.2,clinTF=True): 
    T1 = time.time()
    highestN = 1000
    
    #sensitivity = 1
    FAR = 0
    sensLIST = [0.5,1]
    FARlist = [0,2]
    
    metric_type= 'MPC'
    Nlist = np.arange(280,800,10)  
    figname = 'oneExample.jpg'

    
    fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5,5))        
    
    #fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,5))        
    #for ic,clinTF in enumerate(tqdm([True,False],desc='Top level clinTF')):
    #    doThisIter(ax[ic],highestN,clinTF,Nlist,FARlist,sensLIST,numCPUs,REPS,DRG,metric_type)
    clinTF=True
    doThisIter(ax,highestN,clinTF,Nlist,FARlist,sensLIST,numCPUs,REPS,DRG,metric_type)
    
       
    #plt.savefig(figname,dpi=300)
    plt.show()    
    
    print(f'Time = {time.time()-T1}')
    
def doThisIter(thisax,highestN,clinTF,Nlist,FARlist,sensLIST,numCPUs,REPS,DRG,metric_type):
    # find the highest N needed from Nlist and make a heatmap of it
    L = len(FARlist)*len(sensLIST)
    k = pd.DataFrame(np.zeros((L,3)),columns=['FAR','sensitivity','N'])
    ind = 0

    finalN = highestN
    for fi,FAR in enumerate(tqdm(FARlist,desc='FAR',leave=False)):
        for si,sensitivity in enumerate(tqdm(sensLIST,desc='sens',leave=False)):
            for Ni in trange(len(Nlist),desc='N',leave=False):
                N = Nlist[Ni]
                p = get_pow_kind_full(N,numCPUs,REPS,DRG,sensitivity,FAR,clinTF,metric_type)
                if p>0.9:
                    finalN=N
                    break
            k.iloc[ind,:] = [FAR,sensitivity,finalN]
            ind+=1
    k['N'] = k['N'].astype('int')
    thispow = k.pivot('FAR','sensitivity','N')
    sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=thisax)
    thisax.set_title(f'Power Metric={metric_type} clinTF={clinTF}')    

