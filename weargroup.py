from turtle import down
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
from numpy.random import default_rng
import scipy.stats as stats

#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

def add_FAR(x,FAR):
    # INPUTS:
    #   x - the diary
    #   FAR - the rate of false alarms PER SAMPLE
    # OUTPUTS:
    #   newx - the diary with extra alarms in there

    # we assume FAR is in terms of SAMPLES    
    if FAR==0:
        return x
    else:
        sampRATE=1
        
        randNums = np.random.randn(len(x))
        added_sz = FAR*sampRATE + (FAR*sampRATE/2)*randNums
        added_sz = np.round(np.random.randn(len(x))+FAR).astype('int')
        
        return x + added_sz

def make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=1):
    # INPUTS:
    #  sampRATE = samples per day
    #  howmanydays = how many days to generate
    #  makeOBS =[default False] True: make observed_dairy, False: don't    
    #  downsample_rate =[default 1] - downsample output by how much?
    
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
    obs_sensitivity = 0.5       # Elgar 2018
    obs_FAR = 0.0               # ?? "we've got to start somewhere"
    
    # generate a monthly seizure frequency that is realistic
    mSF = get_mSF( requested_msf=-1 )
    
    # increase true SF to account for under-reporting
    mSF /= obs_sensitivity
    
    # decrease true SF to account for over-reporting
    mSF /= (1 + obs_FAR) 

    # account for difference intracranial vs clinical
    esz_vs_all_sz = esz_vs_all_sz_std * np.random.randn() + esz_vs_all_sz_mean
    temp_sz = np.max([.03,esz_vs_all_sz])       # error check
    esz_vs_all_sz = np.min([temp_sz,.72])       # error check
    #  Neurovista data has limits 0.3 - 0.72
    
    # eFactor = multiply it by # clin szs and give # of e-seizres+clini szs
    eFactor =  (1 / (1 - esz_vs_all_sz) )    
    mSF_all = mSF * eFactor
    
    # generate true electrographic diary (which includes true clin szs too)
    true_e_diary = simulator_base(sampRATE=sampRATE,number_of_days=howmanydays,defaultSeizureFreq=mSF_all)
    
    # remove a percentage of the complete set of seizures to get the true clinical set
    efficacy = 1/eFactor
    true_clin_diary = applyDrug(efficacy=efficacy,x=true_e_diary,baseline=0)
    
    # downsample true diaries if requested (ie downsample_rate>1)
    if downsample_rate>1:
        true_e_diary = downsample(true_e_diary,downsample_rate)
        true_clin_diary = downsample(true_clin_diary,downsample_rate)
        
    if makeOBS:
        # calculate the observed clinical seizures
        observed_clin_diary = remove_and_add(true_clin_diary,obs_sensitivity,obs_FAR,sampRATE)
        # if requested, this is also needed
        if downsample_rate>1:
            observed_clin_diary = downsample(observed_clin_diary,downsample_rate)
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


## NEWSTUFF

def build_many_cases(numCPUs=9,trials=10000,maxN=1000,baseline=2,test=3,DRG=0.2):
    REPS = trials*maxN
    trialDur = baseline+test
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        EandC = par(delayed(make_one_pt_months)(trialDur) for _ in trange(REPS))
    arr = np.array(EandC,dtype=int)
    EandC = np.reshape(arr,newshape=(trials,maxN,trialDur*2))
    E = EandC[:,:,0:trialDur]
    E = apply_drug_bulk(E,maxN,efficacy=DRG,baseline=baseline)
    C = EandC[:,:,trialDur:]
    C = apply_drug_bulk(C,maxN,efficacy=DRG,baseline=baseline)
    return E,C
    
    
def apply_drug_bulk(X,N,efficacy,baseline):
    drugGroup = int(N/2)
    X2 = X.copy()
    X2[:,drugGroup:,baseline:] = np.round(X[:,drugGroup:,baseline:]*(1-efficacy))
    return X2
    
def make_one_pt_months(trialDur):
    sampRATE = 1
    howmanydays = 28*trialDur
    ediaryM,trueclinM = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=28)
    
    return np.concatenate([ediaryM,trueclinM])

def getPC_months(trialData,baseline,test):
    B = np.sum(trialData[:,:baseline],axis=1) / (baseline)
    T = np.sum(trialData[:,baseline:],axis=1) / (test)
    PC = 100*np.divide(B-T,B)
    return PC

def drop_add_seizures_prob(xdatai,sens,FAR,maxN,trialDur):
    sampRATE=28
    cutdown = 0.2
    delt = np.zeros((maxN,trialDur))
    randNums = np.random.randn(maxN,trialDur)
    Rss = np.zeros((maxN,trialDur))
    for J in range(maxN):
        for K in range(trialDur):
            Rs = np.random.rand(xdatai[J,K])
            Rss[J,K] = np.sum(Rs>sens)
            #delt[J,K] = -np.sum(Rs>sens) + FAR*sampRATE + (FAR*sampRATE/2)*randNums[J,K]
            #delt[J,K] = np.max([delt[J,K],-xdatai[J,K]])
    #delt = -Rss + FAR*sampRATE + (FAR*sampRATE/4)*randNums
    #delt = -Rss + FAR*sampRATE + sampRATE*randNums
    delt = -Rss + FAR*sampRATE + sampRATE*randNums * cutdown
    
    delt = np.max(np.concatenate([[-xdatai],[delt]]),axis=0)
    return np.round(delt).astype('int')

def savePCs(theData,baseline,test,FAR,sens,numCPUs,fname):
    xdata = theData.copy()
    #account for sensitivity
    trialDur = baseline+test
    trialCount = xdata.shape[0]
    maxN = xdata.shape[1]
    if (FAR>0 or sens<1):
        # compute a probabistic sensitivity and FAR for catching each seizure
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            tempx = par(delayed(drop_add_seizures_prob)(xdata[ti,:,:],sens,FAR,maxN,trialDur) for ti in trange(trialCount,desc='adding FAR/sense',leave=False))
        deltx = np.array(tempx,dtype=int)
        xdata += deltx


    B = np.sum(xdata[:,:,:baseline],axis=2) / baseline
    T = np.sum(xdata[:,:,baseline:],axis=2) / test
    np.seterr(divide='ignore',invalid='ignore')
    PC = 100*np.divide((B-T),B)
    np.save(fname,PC)
    return

def buildAllTrials_v2(fname,trialCount,baseline,test,maxN,numCPUs,maxPOW=0.9):
    
    PC = np.load(fname)
    if maxPOW == 0.9:    
        maxMPC = 1930
        maxRR50= 2650
    elif maxPOW == 0.8:
        maxMPC = 1460
        maxRR50 = 2000
    else:
        maxMPC = maxN
        maxRR50 = maxN
        
    rng = default_rng()
    trialDur = baseline+test
    
    Nlist_mpc = np.arange(100,maxN,10)
    finalN_mpc = maxN
    MPC = [0,0]
    pow_list = np.zeros(len(Nlist_mpc))
    #for ni,N in enumerate(tqdm(Nlist_mpc,desc='N')):
    for ni,N in enumerate(Nlist_mpc):
    
        pow = get_pows_given_n(maxN,N,trialCount,PC,mpcTF=True,rng=rng,numCPUs=numCPUs)
        pow_list[ni] = pow
        if pow>maxPOW:
            finalN_mpc = N
            break
    
    Nlist_rr50 = np.arange(900,maxN,10)
    finalN_rr50 = maxN
    pow_list = np.zeros(len(Nlist_rr50))
    #for ni,N in enumerate(tqdm(Nlist_rr50,desc='N')):
    for ni,N in enumerate(Nlist_rr50):
        pow = get_pows_given_n(maxN,N,trialCount,PC,mpcTF=False,rng=rng,numCPUs=numCPUs)
        pow_list[ni] = pow
        if pow>maxPOW:
            finalN_rr50 = N
            break
    
    #plt.plot(Nlist_rr50,pow_list,'.-')
    #plt.show()
    return 100*(finalN_mpc/maxMPC),100*(finalN_rr50/maxRR50)

    
    
        
    
def get_pows_given_n(maxN,N,trialCount,PC,mpcTF,rng,numCPUs=9):    
    # (when mpcTF=False, PC = RR50)
    
    # which patients do we want?
    maxN2 = int(maxN/2)
    N2 = int(N/2)
    #inds = rng.permutation(np.arange(0,maxN2))
    #inds0 = inds[0:N2]
    inds0 = np.arange(0,N2)
    #inds = rng.permutation(np.arange(maxN2,maxN))
    #inds1 = inds[0:N2]
    inds1 = np.arange(maxN2,(maxN2+N2))
    #if mpcTF==False:
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        p = par(delayed(get_pow_sub1_given_n)(PC,ti,inds0,inds1,mpcTF) for ti in range(trialCount))
    plist = np.array(p,dtype=float)
    #else:        
    #    [_, plist] = stats.ranksums(PC[:,inds0], PC[:,inds1])

    pow=np.nansum(plist<0.05) / trialCount
        
    return pow

def get_pow_sub1_given_n(PC,ti,inds0,inds1,mpcTF):
    PC_pcb = PC[ti,inds0]
    PC_drg = PC[ti,inds1]
    #PC_pcb = PC_pcb[~np.isnan(PC_pcb)]
    #PC_drg = PC_drg[~np.isnan(PC_drg)]
    if mpcTF==True:
        p = calculate_MPC_p_value_nan(PC_pcb,PC_drg)
    else:
        #p = calculate_fisher_exact_p_value_from_responderTF(PC_pcb,PC_drg)
        p = calculate_fisher_exact_or_chi2_p_value(PC_pcb,PC_drg)
        #p = calculate_fisher_exact_p_value(PC_pcb,PC_drg)
    return p

def calculate_MPC_p_value_nan(placebo_arm_percent_changes,
                                     drug_arm_percent_changes):

    placebo_arm_percent_changes = placebo_arm_percent_changes[~np.isnan(placebo_arm_percent_changes)]
    drug_arm_percent_changes = drug_arm_percent_changes[~np.isnan(drug_arm_percent_changes)]
    # Mann_Whitney_U test
    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, drug_arm_percent_changes)

    return MPC_p_value

def calculate_fisher_exact_or_chi2_p_value(placebo_arm_percent_changes,
                                   drug_arm_percent_changes):

    num_placebo_arm_responders     = np.nansum(placebo_arm_percent_changes > 50)
    num_drug_arm_responders        = np.nansum(drug_arm_percent_changes    > 50)
    num_placebo_arm_non_responders = len(placebo_arm_percent_changes) - num_placebo_arm_responders
    num_drug_arm_non_responders    = len(drug_arm_percent_changes)    - num_drug_arm_responders

    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])
    #if np.all(table>5):
    #    [_,RR50_p_value,_,_] = stats.chi2_contingency(table)
    #else:
    [_, RR50_p_value] = stats.fisher_exact(table)

    return RR50_p_value



def premake_clin(O,sensLIST,FARlist,maxTrials,maxN,clinTF,numCPUs=9):
    # naganur 2022 automated seizure detection with noninvasive wearable devices...
    # FAR 1.7-2.6, sens .85-96
    baseline=2
    test=3
    if clinTF==True:
        fnprefix = f'O-{maxTrials}-by-{maxN}'
    else:
        fnprefix = f'E-{maxTrials}-by-{maxN}'
        
    for sens in tqdm(sensLIST,desc='sens'):
        for FAR in tqdm(FARlist,desc='FAR',leave=False):
            fname = f'{fnprefix}-sens{sens}-FAR{FAR}.npy'
            savePCs(O,baseline,test,FAR,sens,numCPUs,fname)

            
def tryAllClin(maxN,maxTrials,sensLIST,FARlist,clinTF,fn_csv,maxPOW=0.9,numCPUs=9):
    # naganur 2022 automated seizure detection with noninvasive wearable devices...
    # FAR 1.7-2.6, sens .85-96
    baseline=2
    test=3
    if clinTF==True:
        fnprefix = f'O-{maxTrials}-by-{maxN}'
    else:
        fnprefix = f'E-{maxTrials}-by-{maxN}'
        
    iter = 0
    for sens in tqdm(sensLIST,desc='sens'):
        for FAR in tqdm(FARlist,desc='FAR',leave=False):
            fname = f'{fnprefix}-sens{sens}-FAR{FAR}.npy'
            #savePCs(O,baseline,test,FAR,sens,numCPUs,fname)
            finalN_m,finalN_r = buildAllTrials_v2(fname,maxTrials,baseline,test,maxN,numCPUs,maxPOW=maxPOW)

            #finaN_m,finalN_r = buildAllTrials(O,trialCount=maxTrials,baseline=baseline,test=test,maxN=maxN,sens=sens,FAR=FAR,numCPUs=numCPUs,maxPOW=maxPOW)
            df = pd.DataFrame({'sense':[sens],'FAR':[FAR],'N_m':[finalN_m],'N_r':[finalN_r]})
            if iter>0:
                k = pd.concat([df,k])
            else:
                k = df
            iter+=1
            
    #print(bigDF)

    k['N_m'] = k['N_m'].astype('int')
    k['N_r'] = k['N_r'].astype('int')
    k.to_csv(fn_csv,index=False)
    
    fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))     

    metric_type='MPC'
    thispow = k.pivot('FAR','sense','N_m')
    sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=ax[0],vmin=0,vmax=100)
    ax[0].set_title(f'Power Metric={metric_type} clinTF={clinTF}')    
    metric_type='RR50'
    thispow = k.pivot('FAR','sense','N_r')
    sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=ax[1],vmin=0,vmax=100)
    ax[1].set_title(f'Power Metric={metric_type} clinTF={clinTF}')   
    plt.show()
    
def do_a_clin_set(sensLIST,FARlist,csvFN,doPremake=True,filePREFIX='Patients'):

    maxTrials = 10000
    maxN = 4000
    print(' E...')
    E = np.load(f'e{filePREFIX}{maxTrials}-by-{maxN}.npy')
    print(' O...')
    O = np.load(f'o{filePREFIX}{maxTrials}-by-{maxN}.npy')
    print('ready.')

    maxPOW = 0.9
    clinTF = False
    if doPremake:
        premake_clin(E,sensLIST,FARlist,maxTrials,maxN,clinTF=clinTF,numCPUs=9)
    tryAllClin(maxN,maxTrials,sensLIST,FARlist,clinTF=clinTF,fn_csv=f'{clinTF}_{csvFN}.csv',maxPOW=maxPOW)
    clinTF = True
    if doPremake:
        premake_clin(O,sensLIST,FARlist,maxTrials,maxN,clinTF=clinTF,numCPUs=9)
    tryAllClin(maxN,maxTrials,sensLIST,FARlist,clinTF=clinTF,fn_csv=f'{clinTF}_{csvFN}.csv',maxPOW=maxPOW)

## Tools for PART 2
# run for 10 years per patient x 10000 patients
# Review 3 month seizure history= rate[1]
# Compare to prior 3 months seizure history= rate[0]
# if rate[1] = 0 or rate[1] =< (½) rate[0], NO CHANGE
# If NO CHANGE for 2 years, remove one drug
# if rate[1] > (½) rate[0], add med with 20% efficacy - but only do it 80% of the time because of conservative clinician
# CORRECT treatment = the above algorithm based on “true clinical” diary. Compare CORRECT to others:
# % patients that get PERFECT match = P1
# % of patients that never get less meds then CORRECT = P2
# P3  = P2-P1 → % patients that get TOO MUCH meds
# P4 = 100-P2 —> % patients that get NOT ENOUGH meds
# RATE of PERFECT clinical decisions
# RATE of OVERDRUGGING
# RATE of UNDERDRUGGING
# have some epsilon, like 10% in either direction





def simulate_clinic(sensLIST,FARlist,N=10000,yrs=10,numCPUs=9):
    clinic_interval = 30*3     # 3 months between clinic visits
    yrs = 10
    
    # generate all patients first
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        tempx = par(delayed(build_mypt)(yrs,clinic_interval) for _ in trange(N,desc='generate patient'))
        mypt = np.array(tempx,dtype=int)
    # mypt = N x 2 x samples (one sample per 3 months for 10 yrs)
    #      mypt[:,0,:] = true_e
    #      mypt[:,1,: = true_clin
    for sens in tqdm(sensLIST,desc='sensitivity'):
        for FAR in tqdm(FARlist,desc='FAR'):
            myptSENSOR = compute_sensor(mypt,FAR,sens)
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                tempx = par(delayed(do_one_pt_clinic_rates)(mypt[ti,:,:],sens,FAR) for ti in trange(N,desc='run patient'))
                deltx = np.array(tempx,dtype=float)


def do_one_pt_clinic_rates(mypt,sens,FAR):
    sampRATE=30*3
    
    ptDUR = mypt.shape[1]
    
    delt = np.zeros(ptDUR)
    randNums = np.random.randn(ptDUR)
    Rss = np.zeros(ptDUR)
    for K in range(ptDUR):
        Rs = np.random.rand(mypt[K])
        Rss[K] = np.sum(Rs>sens)
    delt = -Rss + FAR*sampRATE + randNums
    delt = np.max(np.concatenate([[-mypt],[delt]]),axis=0)
    return np.round(delt).astype('int')

    decisionList_clin,decisionList_e,decisionList_true = simulate_1pt_in_clinic(mypt)
    rate_clin, overall_clin = compute_this_clinical_rate(decisionList_clin,decisionList_true)
    rate_e, overall_e = compute_this_clinical_rate(decisionList_e,decisionList_true)
### NOT DONE
    
    return rate_clin,overall_clin,rate_e,overall_e

def simulate_1pt_in_clinic(mypt):
    e_pt = mypt[0,:]
    c_pt = mypt[1,:]
    
    return decisionList_clin,decisionList_e,decisionList_true

def build_mypt(yrs,clinic_interval):
    sampRATE = 6
    howmanydays = yrs*30*12
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=sampRATE*clinic_interval)

    return np.concatenate([[true_e_diary], [true_clin_diary]])