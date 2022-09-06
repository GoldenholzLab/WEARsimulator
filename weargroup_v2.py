import numpy as np
from realSim import get_mSF, simulator_base,downsample
from trialSimulator import getPC
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

#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  


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
        x2 = [  applyDrugOneSample(x[iter],efficacy) for iter in range(baseline,L) ]    
        
        return x2
    else:
        # DO NOITHING for efficacy = 0
        return x

def applyDrugOneSample(samp,efficacy):
    return samp - np.sum(np.random.random(int(samp))<efficacy)

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
    
    Xs = add_sens_and_FAR(sensitivity,FAR,X,downsampleRATE,inflater)

    return Xs

def add_sens_and_FAR(sensitivity,FAR,X,downsampleRATE,inflater=2/3):
     # add sensitivity
    if sensitivity<1:
        Xs = applyDrug(efficacy=(1-sensitivity),x=X,baseline=0)
    else:
        Xs = X.copy()
    # add FAR
    if FAR>0:
        L = len(X)
        downsampleRATEhalf = 0.5*downsampleRATE
        Xadder = np.zeros(L).astype('int')
        for i in range(L):
            x = np.random.random(downsampleRATE)
            zeroMean = FAR*inflater*(np.sum(x) - downsampleRATEhalf)
            Xadder[i] = np.round(downsampleRATE*FAR + zeroMean).astype('int')
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
# r1 = RATE of PERFECT clinical decisions
# r2 = RATE of OVERDRUGGING
# r3 = RATE of UNDERDRUGGING
# have some epsilon, like 10% in either direction


### Model of sz freedom (Chen et al 2018)
# 1rst: 46%
# 2nd: 28%
# 3rd: 24%
# 4th: 15%
# 5th: 14%
# 6th: 14%


def show_me_one(sens,FAR,N,yrs=10,numCPUs=9):
    clinic_interval = 30*3     # 3 months between clinic visits
    yrs = 10
    episilon_rate = 12       # 3 clinic visit can mess up
    # generate all patients first
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        tempx = par(delayed(build_mypt)(yrs,clinic_interval) for _ in range(N))
    mypt = np.array(tempx,dtype=int)
    L = mypt.shape[2]
    #plt.subplot(3,1,1)
    t = np.arange(40) / 4
    #for i in range(N):
        #plt.plot(t,mypt[i,0,:],'b',alpha=0.2)
        #plt.plot(t,mypt[i,1,:],'r',alpha=0.2)
    plt.plot(t,np.median(mypt[:,0,:],axis=0),'-b',label='e')
    plt.plot(t,np.median(mypt[:,1,:],axis=0),'-r',label='clin')
    #plt.legend()
    #plt.show()
    
    #plt.subplot(3,1,2)
    Xe = np.zeros((N,40))
    Xc = np.zeros((N,40))
    Xe_sim = np.zeros((N,40))
    Xc_sim = np.zeros((N,40))
    decisionList_clin = np.zeros((N,40))
    decisionList_e = np.zeros((N,40))
    decisionList_true = np.zeros((N,40))
    for i in range(N):
        Xe[i,:] = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[i,0,:],downsampleRATE=clinic_interval,inflater=2/3)
        Xc[i,:] = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[i,1,:],downsampleRATE=clinic_interval,inflater=2/3)
        #plt.plot(t,Xe[i,:],'b',alpha=0.2)
        #plt.plot(t,Xc[i,:],'r',alpha=0.2)
        decisionList_clin[i,:],decisionList_e[i,:],decisionList_true[i,:],trueX_sim,Xe_sim[i,:],Xc_sim[i,:] = simulate_1pt_in_clinic(mypt[i,1,:],Xe[i,:],Xc[i,:])
    
    #plt.plot(t,np.median(Xe,axis=0),':b',label='e-device')
    #plt.plot(t,np.median(Xc,axis=0),':r',label='clin-device')
    plt.plot(t,np.mean(Xe,axis=0),':b',label='e-device')
    plt.plot(t,np.mean(Xc,axis=0),':r',label='clin-device')
    #plt.title('device applied')
    #plt.legend()
    #plt.show()
    
    #plt.subplot(3,1,3)
    #plt.plot(t,np.median(Xe_sim,axis=0),'--b',label='e-device-meds')
    #plt.plot(t,np.median(Xc_sim,axis=0),'--r',label='clin-device-meds')
    plt.plot(t,np.mean(Xe_sim,axis=0),'--b',label='e-device-meds')
    plt.plot(t,np.mean(Xc_sim,axis=0),'--r',label='clin-device-meds')
    
    plt.title('device applied med applied')
    plt.legend()
    plt.show()
    
    #plt.plot(t,np.median(decisionList_true,axis=0),label='true')
    #plt.plot(t,np.median(decisionList_e,axis=0),label='e')
    #plt.plot(t,np.median(decisionList_clin,axis=0),label='clin')
        
    plt.plot(t,np.mean(decisionList_true,axis=0),'r',label='true')
    plt.plot(t,np.mean(decisionList_e,axis=0),'b',label='e')
    plt.plot(t,np.mean(decisionList_clin,axis=0),'k',label='clin')
    plt.plot(t,np.mean(decisionList_true,axis=0)+np.std(decisionList_true),'r',alpha=0.4)
    plt.plot(t,np.mean(decisionList_true,axis=0)-np.std(decisionList_true),'r',alpha=0.4)
    plt.plot(t,np.mean(decisionList_e,axis=0)+np.std(decisionList_e),'b',alpha=0.4)
    plt.plot(t,np.mean(decisionList_e,axis=0)-np.std(decisionList_e),'b',alpha=0.4)
    plt.plot(t,np.mean(decisionList_clin,axis=0)+np.std(decisionList_clin),'k',alpha=0.4)
    plt.plot(t,np.mean(decisionList_clin,axis=0)-np.std(decisionList_clin),'k',alpha=0.4)


    #plt.boxplot(decisionList_true)
    plt.legend()
    plt.title('med counts')
    plt.show()

    
def simulate_clinic(sensLIST,FARlist,N=10000,yrs=10,numCPUs=9):
    clinic_interval = 30*3     # 3 months between clinic visits
    yrs = 10
    episilon_rate = 12       # 3 clinic visit can mess up
    # generate all patients first
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        tempx = par(delayed(build_mypt)(yrs,clinic_interval) for _ in range(N))
    mypt = np.array(tempx,dtype=int)
    L = mypt.shape[2]

    df = pd.DataFrame()
    for sens in tqdm(sensLIST,desc='sensitivity'):
        for FAR in tqdm(FARlist,desc='FAR'):
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                tempx = par(delayed(do_one_pt_clinic_rates)(mypt[ti,:,:],clinic_interval,sens,FAR) for ti in range(N))
            pset = np.array(tempx,dtype=float)
            print(pset)
            p1c = np.mean(pset[:,0]>(L-episilon_rate))
            p2c = np.mean((pset[:,1]+pset[:,0])>episilon_rate)
            r1c = np.mean(pset[:,0]) / L
            r2c = np.mean(pset[:,1]) / L
            p1e = np.mean(pset[:,2]>(L-episilon_rate))
            p2e = np.mean(pset[:,3]>episilon_rate)
            r1e = np.mean(pset[:,2]) / L
            r2e = np.mean(pset[:,3]) / L
            
            df2 = pd.DataFrame({'sens':[sens],'FAR':[FAR],'p1c':[p1c],'p2c':[p2c],'r1c':[r1c],'r2c':[r2c],
                                'p1e':[p1e],'p2e':[p2e],'r1e':[r1e],'r2e':[r2e]})
            df = pd.concat([df,df2])            
    
    pd.set_option('display.precision', 3)
    display(df)
    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,10))     
    d3 = df.copy()
    d3=d3.drop(columns=['p1e','p2e','r1e','r2e','p2c','r1c','r2c'])
    thispow = df.pivot('FAR','sens','p1c')
    
    sns.heatmap(thispow, annot=True,linewidths=0.5,vmin=0,vmax=1,ax=ax[0,0])
    ax[0,0].set_title(f'p1c - % patients that get PERFECT match CLINICAL') 
    d3 = df.copy()
    d3=d3.drop(columns=['p2e','r1e','r2e','p1c','p2c','r1c','r2c'])
    thispow = df.pivot('FAR','sens','p1e')
    
    sns.heatmap(thispow, annot=True,linewidths=0.5,vmin=0,vmax=1,ax=ax[0,1])
    ax[0,1].set_title(f'p1e - % patients that get PERFECT match ELECTROGRAPHIC') 
    d3 = df.copy()
    d3=d3.drop(columns=['p1e','p2e','r1e','r2e','p1c','r1c','r2c'])
    thispow = df.pivot('FAR','sens','p2c')
    
    sns.heatmap(thispow, annot=True,linewidths=0.5,vmin=0,vmax=1,ax=ax[1,0])
    ax[1,0].set_title(f'p2c - % ever get less meds then CORRECT clin') 
    d3 = df.copy()
    d3=d3.drop(columns=['p1e','r1e','r2e','p1c','p2c','r1c','r2c'])
    thispow = df.pivot('FAR','sens','p2e')
    
    sns.heatmap(thispow, annot=True,linewidths=0.5,vmin=0,vmax=1,ax=ax[1,1])
    ax[1,1].set_title(f'p2e -  % ever get less meds then CORRECT electr') 
    plt.show()
    

       
    

def do_one_pt_clinic_rates(mypt,clinic_interval,sens,FAR):
    
    Xe = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[0,:],downsampleRATE=clinic_interval,inflater=2/3)
    Xc = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[1,:],downsampleRATE=clinic_interval,inflater=2/3)
    
    decisionList_clin,decisionList_e,decisionList_true,_,_,_ = simulate_1pt_in_clinic(mypt[1,:],Xe,Xc)
    r1c,r2c = compute_this_clinical_rate(decisionList_clin,decisionList_true)
    r1e,r2e = compute_this_clinical_rate(decisionList_e,decisionList_true)
    
    return np.array([r1c,r2c,r1e,r2e])

def compute_this_clinical_rate(decisionList_x,decisionList_true):
    # r1 = number of correct decisions
    # r2 = number of over-drugging decisions
    #r1 = np.sum(decisionList_true==decisionList_true)
    r1 = np.sum(decisionList_x==decisionList_true)
    r2 = np.sum(decisionList_x>decisionList_true)
    return r1,r2


def simulate_1pt_in_clinic(myptT,Xe,Xc):    
    L = len(myptT)
    decisionList_true, newMypt = do_decisions(myptT,L)
    decisionList_e, newXe = do_decisions(Xe,L)
    decisionList_c, newXc = do_decisions(Xc,L)
    
    return decisionList_c,decisionList_e,decisionList_true,newMypt,newXe,newXc

def do_decisions(myX,L):
    X = myX.copy()
    addChance = .5
    twoyears = 8        # 8 visits 3 months apart = 2 years
    drugStrength = 0.2
    decisionList = np.zeros(L).astype('int')
    szFree = np.zeros(L).astype('int')
    nochangeCounter = 0
    maxMEDS = 6
    # the first 3 months will have a decision of do nothing, so start on the second visit.
    for i in range(1,L):
        # first, apply the previous decision to this sample
        if szFree[i] == 1:
            X[i] = 0
        else:
            for drugNum in range(decisionList[i-1]):
                X[i] = applyDrugOneSample(samp=X[i],efficacy=drugStrength)

        # now make this clinic's decision based on the result        
        if X[i]<= (0.5 * X[i-1]):
            # no change condition
            decisionList[i] = decisionList[i-1]
            # check if 2 yrs sz free
            nochangeCounter += 1
            if nochangeCounter==twoyears:
                # nothing bad in 2 years, decrease med
                decisionList[i] = np.max([ (decisionList[i-1] - 1) ,0])
                nochangeCounter = 0
        else:
            # add med condition... 
            # probabilistically add a med
            coinFlip = np.random.random()<addChance
            if coinFlip==True:
                # now actually add a med
                decisionList[i] = decisionList[i-1]+1        
                decisionList[i] = np.min([decisionList[i],maxMEDS])
                nochangeCounter = 0
                ### Model of sz freedom (Chen et al 2018)
                # 1rst: 46%
                # 2nd: 28%
                # 3rd: 24%
                # 4th: 15%
                # 5th: 14%
                # 6th: 14%
                if decisionList[i]==1:
                    changeSZFREE = .46
                elif decisionList[i]==2:
                    changeSZFREE = .28
                elif decisionList[i]==3:
                    changeSZFREE = .24
                elif decisionList[i]==4:
                    changeSZFREE = .15
                elif decisionList[i]==5:
                    changeSZFREE = .14
                elif decisionList[i]==6:
                    changeSZFREE = .14
                else:
                    changeSZFREE = 0
                if np.random.random() < changeSZFREE:
                    szFree[i:] = 1
                    
    return decisionList,X
                
def build_mypt(yrs,clinic_interval):
    sampRATE = 6
    howmanydays = yrs*30*12
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=sampRATE*clinic_interval)

    return np.concatenate([[true_e_diary], [true_clin_diary]])