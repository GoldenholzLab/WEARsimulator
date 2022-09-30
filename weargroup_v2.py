import numpy as np
from realSim import get_mSF, simulator_base,downsample
from trialSimulator import getPC
from trialSimulator import calculate_fisher_exact_p_value, calculate_MPC_p_value
from weargroup import make_multi_diaries
from joblib import Parallel, delayed
#from tqdm.notebook import trange, tqdm
from tqdm import trange, tqdm
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
        
        return np.array(x2).astype('int')
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


def add_sens_and_FAR_onesamp(sensitivity,FAR,X,downsampleRATE,inflater=2/3):
    if sensitivity<1:
        Xs = applyDrugOneSample(samp=X,efficacy=(1-sensitivity))
    else:
        Xs = X
    if FAR>0:
        downsampleRATEhalf = 0.5*downsampleRATE
        Xadder = 0
        x = np.random.random(downsampleRATE)
        zeroMean = FAR*inflater*(np.sum(x) - downsampleRATEhalf)
        Xadder = np.round(downsampleRATE*FAR + zeroMean).astype('int')
        if Xadder<0:
            Xadder = 0
        Xs += Xadder
    return int(Xs)

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

def do_some_sets(inflater=2/3,sLIST = [1,.9,.8],fLIST= [ 0, 0.05, 0.1],N=10000,NOFIG=False,fn='',clinTF=True):
    if NOFIG==False:
        plt.subplots(3,3,sharex=True,sharey=True,figsize=(12,12))
    counter = 0
    df = pd.DataFrame()
    
    for si,sens in enumerate(tqdm(sLIST,desc='sensitivity')):
        for fi,FAR in enumerate(tqdm(fLIST,desc='FAR',leave=False)):
    
    #for si,sens in enumerate(sLIST):
    #    for fi,FAR in enumerate(fLIST):
    
            counter+=1
            if NOFIG==False:
                plt.subplot(3,3,counter)
            szfree, drugCounts,szCounts = show_me_set(sens=sens,FAR=FAR,N=N,clinTF=clinTF,showTF=False,noFig=NOFIG,inflater=inflater)
            df = pd.concat([df,pd.DataFrame({'sens':[sens],'FAR':[FAR],'szfree':[szfree],'meanDrug':[np.mean(drugCounts)],'meanSz':[np.mean(szCounts)]})])
    
    if NOFIG==False:
        plt.show()
    print(df)
    if fn != '':
        df.to_csv(fn,index=False)
    return df

def show_me_set(sens,FAR,N,clinTF,numCPUs=9,showTF=True,noFig=False,inflater=2/3):
    # define some constants
    clinic_interval = 30*3     # 3 months between clinic visits
    yrs = 10
    L = int(10*12*30 / clinic_interval)
    
    # run each patient
    if numCPUs>1:
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(sim1clinic)(sens,FAR,clinic_interval,clinTF,L,inflater) for _ in range(N))
    else:
        temp = [ sim1clinic(sens,FAR,clinic_interval,clinTF,L,inflater) for _ in trange(N)]

    bigX = np.array(temp,dtype=int)
    trueCount = bigX[:,0]
    sensorCount = bigX[:,1]
    drugCountSensor = bigX[:,2]
    szFree =bigX[:,3]
    
    toMonths = L * 3
    if noFig==False:
        BINS = [np.arange(15),np.linspace(0,6,24)]
        plt.hist2d(trueCount/toMonths,drugCountSensor/toMonths,bins=BINS)
        plt.xlabel('True clinical seizures')
        plt.ylabel('Drug months')
        plt.title(f'Sens={sens} FAR ={FAR}')
    if showTF==True:
        plt.show()
        print(f'FAR = {FAR} sensitivity = {sens} Sz Free = {np.mean(szFree)}')
    else:
        return np.mean(szFree),trueCount/toMonths,drugCountSensor/toMonths

def sim1clinic(sens,FAR,clinic_interval,clinTF,L,inflater):
    # simulate 1 patient all the way through
    
    # constants
    yrs=10
    addChance = .8
    twoyears = 8        # 8 visits 3 months apart = 2 years...
    drugStrengthLO = 0.1
    drugStrengthHI = 0.2
    maxMEDS = 6.5
    doDISCOUNT = True

    ### Model of sz freedom (Chen et al 2018) (note we use percentage of total cohort % values)
    ### also Brodie et al 2012
    r = np.random.random()
    patternCutoffs = np.cumsum([0.37,0.22,0.16,0.25])
    if r<patternCutoffs[0]:
        # early lasting sz freedom
        patternABCD = 0
    elif r<patternCutoffs[1]:
        # delayed lasting sz freedom
        patternABCD = 1
    elif r<patternCutoffs[2]:
        # fluctuating 1yr sz freedom
        patternABCD = 2
    else:
        # no sz freedom
        patternABCD = 3
        
    successLIST = [0,.46,.28,.24,.15,0.14,0.14,0]
    #successLIST = [0,.72,.18,.07,.02,0.01,0.01,0]
    
    # First make 1 patient true_e and true_c
    sampRATE = 6
    howmanydays = yrs*30*12
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=sampRATE*clinic_interval)

    if clinTF==True:
        X = true_clin_diary.copy().astype('int')
    else:
        X = true_e_diary.copy().astype('int')
    sensorXdrugged = X.copy()
    trueXdrugged = X.copy()
    
    # Now start looping along
    decisionList = np.zeros(L)
    szFree = np.zeros(L).astype('int')
    nochangeCounter = 0
    drugCount =0
    
    trueXdrugged[0] = X[0]
    sensorXdrugged[0] = add_sens_and_FAR_onesamp(sensitivity=sens,FAR=FAR,X=X[0],downsampleRATE=clinic_interval,inflater=inflater)
    if doDISCOUNT:
        sensorXdrugged[0] = np.max([0,sensorXdrugged[0]-FAR*clinic_interval])
    for i in range(1,L):
        # apply drugs to the sample first
        thisSamp = X[i]
        for Dcount in range(1,7):
            if drugCount==Dcount:
                thisSamp = applyDrugOneSample(samp=thisSamp,efficacy=drugStrengthLO)
            elif drugCount>=(Dcount+0.5):
                thisSamp = applyDrugOneSample(samp=thisSamp,efficacy=drugStrengthHI)
        if szFree[i]==1:
            thisSamp = 0
            
        trueXdrugged[i] = thisSamp
        
        # what does sensor show after that?
        thisI = add_sens_and_FAR_onesamp(sensitivity=sens,FAR=FAR,X=thisSamp,downsampleRATE=clinic_interval,inflater=inflater)
        if doDISCOUNT:
            thisI = np.max([0,thisI-FAR*clinic_interval])

        sensorXdrugged[i] = thisI
        
        lastI = sensorXdrugged[i-1]
        if thisI <= (lastI*0.5):
            # things have improved
            nochangeCounter +=1
            # has this been good for 2 years?
            if nochangeCounter==twoyears:
                # then decrease something
                if np.floor(drugCount)==drugCount:
                    drugCount = np.max([0,drugCount-0.5])
                else:
                    drugCount = np.floor(drugCount)
                
        else:
            # things remain bad
            nochangeCounter=0
            # ready to increase something?
            if np.random.random()<addChance:
                
                if drugCount==0:
                    drugCount=1
                    newDrug=True
                elif drugCount==maxMEDS:
                    # we already are at max. no change
                    drugCount=maxMEDS
                    newDrug=False
                else:
                    drugCount += 0.5
                    newDrug = (np.floor(drugCount)==drugCount)
                if newDrug:
                    # we have added a new drug now
                    # will this make us sz free?
                    dIND = np.floor(drugCount).astype('int')
                    if np.random.random()<successLIST[dIND]:
                        if patternABCD==0:
                            szFree[i:] = 1
                        elif patternABCD==1:
                            # delayed sz free sustained
                            delayT = np.random.randint(4)
                            istart = np.min([i+delayT,L-1])
                            szFree[istart:]
                        else:
                            # fluctuating sz freedom or no szfreedom
                            imax = np.min([i + 4,L])
                            szFree[i:imax] = 1
                            
        
        decisionList[i] = drugCount
        
    vals = [ np.sum(trueXdrugged), np.sum(sensorXdrugged), np.sum(np.floor(decisionList)) , (0+np.any(szFree))]

    return vals

## FOR injury case

def run_injury_case(sens,FAR,N=10000,numCPUs=9,clinTF=True):
    # see review from Beghi 2009
    # there are two different rates for different populations, so we will just get both.
    # the rates are in events per 100 patient years, so we convert to events per 1 patient day.
    injury_rate_per_pt_dayA = 3 / (100*365)
    injury_rate_per_pt_dayB = 300 / (100*365)
    
    df = pd.DataFrame()
    for injuryRate in [injury_rate_per_pt_dayA,injury_rate_per_pt_dayB]:
        if numCPUs>1:
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                temp = par(delayed(simInjuries)(sens,FAR,clinTF,injuryRate) for _ in trange(N))
        #else numCPUs==1:
        #    temp = [ simInjuries(sens,FAR,clinTF,injuryRate,inflater) for _ in trange(N)]
        bigX = np.array(temp,dtype=float)
        noInjury = np.where(bigX[:,1]==0)[0]
        yesInjury = np.where(bigX[:,1]>0)[0]
        detectedFrac = np.mean(bigX[yesInjury,0] / bigX[yesInjury,1])
        temp = pd.DataFrame({'clinTF':[clinTF],'sens':[sens],'FAR':[FAR],
                             'rate':[int(injuryRate*100*365)],
                             'detected%':[int(100*detectedFrac)],'noInjury':[len(noInjury)],'total':[int(np.sum(bigX[:,1]))],
                             'mean':[np.mean(bigX[:,1])]})
        print(temp)
        df = pd.concat([df,temp])
    #print(df)
    return df

def simInjuries(sens,FAR,clinTF,injuryRate):
    
    inflater = 2  # this is bigger than 2/3 because otherwise it will be too small!
    
    # First make 1 patient true_e and true_c
    yrs = 10
    sampRATE = 24*6 # we sample once every 10 minutes
    howmanydays = yrs*365
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=1)
    if clinTF==True:
        thisDiary = true_clin_diary.copy()
    else:
        thisDiary = true_e_diary.copy()

    thisSensor = add_sens_and_FAR(sensitivity=sens,FAR=FAR/sampRATE,X=thisDiary,downsampleRATE=1,inflater=inflater)
    
    numSamples = howmanydays*sampRATE
    plist = np.random.poisson(lam = injuryRate/sampRATE,size=numSamples)
    injuryTime = False
    detectedCount = 0
    injuryCount = 0
    for samp in range(numSamples):
        if plist[samp]>0:
            injuryTime=True
            # now we wait for the next seizure and injure them then. I know - imperfect.
            # if we would require seizure at same time, we might decrease the rate
        if (true_clin_diary[samp]>0) and (injuryTime==True):
            # if there really is a clinical seizure, and it is time for an injury, then do one
            injuryCount+=1
            injuryTime=False
            if thisSensor[samp]>0:
                # I did detect an injury
                detectedCount += 1

    return detectedCount,injuryCount