import numpy as np
from realSim import get_mSF, simulator_base,downsample
from trialSimulator import getPC
from trialSimulator import calculate_fisher_exact_p_value, calculate_MPC_p_value
from weargroup import make_multi_diaries
from joblib import Parallel, delayed
#from tqdm import trange, tqdm
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import seaborn as sns
import pandas as pd
from numpy.random import default_rng
import scipy.stats as stats
from scipy.ndimage import median_filter

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
        x2[baseline:] = [  applyDrugOneSample(x[iter],efficacy) for iter in range(baseline,L) ]    
        
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
            # the following code allows for a small baseline rate change for each sample
            # zeroMean should be a random number with zero mean, but with std dev of 1 if 
            # inflater is 2/3. It could be bigger or smaller standard dev. What this does
            # in practice is make the sensor have more intra-individual variability.
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

def get_a_qualified_patient(minSz,baseline,trialDur,clinTF,sensitivity,FAR,PCB,DRG,useDRG,inflater,accountForFAR,accountForSens):

    isDone = False
    while isDone==False:
        this_pt = get_a_patient(trialDur,baseline,clinTF,sensitivity,FAR,PCB,DRG,useDRG,inflater)
        if accountForSens:
            minSz *= sensitivity
        if accountForFAR:
            isDone = (np.mean(this_pt[0:baseline]) - (FAR*28))>=minSz
        else:
            isDone = np.mean(this_pt[0:baseline])>=minSz
        
    return this_pt

def build_a_trial(N,halfN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater,accountForFAR=True,accountForSens=False):
    trialDur = baseline+test
    
    # build trial data (number of patients by number of months
    trialData = np.zeros((N,trialDur))
    for counter in range(N):
        trialData[counter,:] = get_a_qualified_patient(minSz,baseline,trialDur,clinTF,sensitivity,FAR,PCB,DRG,(counter>=halfN),inflater,accountForFAR,accountForSens)
        
    if accountForFAR:
        trialData = trialData - (FAR*28)
        trialData[trialData<0] = 0

    if accountForSens:
        trialData /= sensitivity
    
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

def findThresh(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,recalc=True,inflater=2,accountForFAR=True,accountForSens=False):
    
    if recalc==True:
        allPCs = buildPCsets(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater,accountForFAR,accountForSens)
    else:
        print('Loading...',end='')
        allPCs = np.load(fn) 
        print('done.')
    threshRR50 = checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF=False)
    threshMPC = checkThresh(allPCs,numCPUs,REPS,maxN,metricMPC_TF=True)
    
    return threshRR50,threshMPC

def buildPCsets(fn,numCPUs,REPS,maxN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater,accountForFAR,accountForSens):
    N=maxN
    halfN=int(maxN/2)
    T1 = time.time()

    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(build_a_trial)(N,halfN,DRG,PCB,minSz,baseline,test,clinTF,sensitivity,FAR,inflater,accountForFAR,accountForSens) for _ in trange(REPS,desc='trials'))
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
    for thisN in range(100,maxN,10):
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

def buildSET_of_N(senseLIST,farLIST,recalc,thiscsv,clinTF=True,REPS=10000,maxN=3000,DRG=0.2,mini=False,inflater=2,accountForFAR=True,accountForSens=False):
    if mini==False:
        numCPUs = 9
        thedir = '/Users/danielgoldenholz/Library/CloudStorage/OneDrive-BethIsraelLaheyHealth/Comprehensive Epilepsy Program & EEG Lab/Research/Goldenholz Lab/wear'
    else:
        numCPUs = 7
        thedir = '/Users/dgodenh/Documents/GitHub/WEARsimulator'
        #thedir = '/Users/dgodenh/OneDrive - Beth Israel Lahey Health/Comprehensive Epilepsy Program & EEG Lab/Research/Goldenholz Lab/wear'
    
    d2 = pd.DataFrame()
    for sensitivity in senseLIST:
        for FAR in farLIST:
            fn = f'{thedir}/PC_{clinTF}_sens{sensitivity}_FAR{FAR}_{REPS}x{maxN}_{inflater}.npy'
            tRR,tMP = findThresh(fn=fn,numCPUs=numCPUs,REPS=REPS,maxN=maxN,DRG=DRG,PCB=0,minSz=4,
                    baseline=2,test=3,clinTF=clinTF,sensitivity=sensitivity,FAR=FAR,recalc=recalc,
                    inflater=inflater,accountForFAR=accountForFAR,accountForSens=accountForSens)
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
    if clinTF==True:
        clinTFtxt = 'Seizures: C'
    else:
        clinTFtxt = 'Seizures: C+E'
    if ax==[]:
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))     
        doShow = True
    else:
        doShow = False
    #maxList = [530, 900]
    maxList = [540,900]
    mlist=['MPC','RR50']
    for mi,metric_type in enumerate(mlist):
        d3 = d2.copy()
        d3[metric_type] = np.round(100*d2[metric_type] / maxList[mi]).astype('int')
        d3=d3.drop(columns=mlist[1-mi])
        thispow = d3.pivot('FAR','sensitivity',metric_type)
        sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=ax[mi],vmin=0,vmax=100)
        ax[mi].set_title(f'Metric={metric_type}, {clinTFtxt}')    
    if doShow==True:
        plt.show()

def make_full_RCT_sets(drawingOn,prefix,sensLIST,farLIST,inflaterLIST=[2/3],mini=False,accountForFAR=True,accountForSens=False):
    theREPS = 10000
    for inflater in inflaterLIST:
        print(f'Inflater = {inflater}')
        if drawingOn:
            fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,10))   
        for ci,clinTF in enumerate([True,False]):        
            fn = f'{prefix}_{clinTF}_inf{inflater}.csv'
            #[0,.05,.1,.5,1,2,10]
            if drawingOn==False:
                buildSET_of_N(sensLIST,farLIST,recalc=True,thiscsv=fn,clinTF=clinTF,REPS=theREPS,maxN=1500,DRG=0.2,
                        mini=mini,inflater=inflater,accountForFAR=accountForFAR,accountForSens=accountForSens)
            else:
                drawGrid(fn,clinTF=clinTF,ax=ax[ci,:])
        
        #if drawingOn:        
        #    plt.show()
    
## Tools for PART 2
# run for 10 years per patient x 10000 patients
# Review 3 month seizure history= rate[1]
# Compare to prior 3 months seizure history= rate[0]
# if rate[1] = 0 or rate[1] =< (½) rate[0], NO CHANGE
# If NO CHANGE for 2 years, remove one drug
# if rate[1] > (½) rate[0], add med with 20% efficacy - but only do it 80% of the time because of conservative clinician
### Model of sz freedom (Chen et al 2018), and patterns of sz freedom (Brodie et al 2012)

def do_some_sets(inflater=2/3,sLIST = [1,.9,.8],fLIST= [ 0, 0.05, 0.1],N=10000,NOFIG=False,fn='',
                 clinTF=True,biggerCSV=False,doDISCOUNT=True,findSteady=False,numCPUs=9,yrs=10):
    if NOFIG==False:
        plt.subplots(3,3,sharex=True,sharey=True,figsize=(12,12))
    counter = 0
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    interval_count = 39 # this is how many intervals from 10 years of 3 month visits gets
    for si,sens in enumerate(tqdm(sLIST,desc='sensitivity')):
        for fi,FAR in enumerate(tqdm(fLIST,desc='FAR',leave=False)):
    
            #for si,sens in enumerate(sLIST):
            #    for fi,FAR in enumerate(fLIST):
    
            counter+=1
            if NOFIG==False:
                plt.subplot(3,3,counter)
            if findSteady==True:
                szfree, szCounts, drugCounts, how_long = show_me_set(sens=sens,FAR=FAR,N=N,clinTF=clinTF,showTF=False,noFig=NOFIG,inflater=inflater,doDISCOUNT=doDISCOUNT,findSteady=findSteady,numCPUs=numCPUs,yrs=yrs)
                df = pd.concat([df,pd.DataFrame({'sens':[sens],'FAR':[FAR],'szfree':[szfree],'meanDrug':[np.median(drugCounts)/interval_count],'meanSz':[np.median(szCounts)/(interval_count*3)],'how_long':[np.median(how_long)]})])
            else:    
                szfree, szCounts, drugCounts = show_me_set(sens=sens,FAR=FAR,N=N,clinTF=clinTF,showTF=False,noFig=NOFIG,inflater=inflater,doDISCOUNT=doDISCOUNT,findSteady=findSteady,numCPUs=numCPUs,yrs=yrs)
                df = pd.concat([df,pd.DataFrame({'sens':[sens],'FAR':[FAR],'szfree':[szfree],'meanDrug':[np.median(drugCounts)/interval_count],'meanSz':[np.median(szCounts)/(interval_count*3)]})])
                how_long = np.zeros(N)
            if biggerCSV==True:
                newd = pd.DataFrame({'sens':[sens]*N,
                                     'FAR':[FAR]*N,
                                     'szfree':[szfree]*N,
                                     'meanDrug':drugCounts/interval_count,
                                     'meanSz':szCounts/(interval_count*3),
                                     'how_long':how_long})
                df2 = pd.concat([df2,newd])
    if NOFIG==False:
        plt.show()
    print(df)
    if fn != '':
        if biggerCSV==True:
            df2.to_csv(fn,index=False)
        else:
            df.to_csv(fn,index=False)
    return df

def plot_the_clinic_sets(f1,f2,fn):
    flist = [0,1/90,3/90,10/90,30/90,1,3]
    # make a pretty graphical representation of the whole thing
    p1 = pd.read_csv(f1)
    f0 = p1[p1['FAR']==0]
    s5 = f0[f0['sens']==0.5]
    xOBS = s5['meanDrug'].values
    yOBS = s5['meanSz'].values
    p2 = pd.read_csv(f2)
    p1['FAR'] = np.round(p1['FAR']*100) / 100
    p2['FAR'] = np.round(p2['FAR']*100) / 100
    p1 = p1.rename(columns={'FAR':'False alarm rate','sens':'Sensitivity'})
    p2 = p2.rename(columns={'FAR':'False alarm rate','sens':'Sensitivity'})
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(2,2,1)
    plt.title('Clinical + Electrographic (sensitivity highlighted)')
    sns.color_palette("bright")
    sns.scatterplot(data=p2,x='meanDrug',y='meanSz',hue='Sensitivity',palette='flare',hue_norm=(0.5,1),
                    size='Sensitivity',size_norm=(.5,1),sizes=(10,100),legend=False)
    #plt.plot(xOBS,yOBS,'xr',markersize=10,alpha=0.5,label='Self-report')
    plt.grid(True)
    #plt.xlabel('Average drugs per month per patient')
    plt.ylabel('Average seizures per month per patient')
    plt.xlabel('')

    plt.subplot(2,2,2)
    plt.title('Clinical (sensitivity highlighted)')
    sns.scatterplot(data=p1,x='meanDrug',y='meanSz',hue='Sensitivity',palette='flare',hue_norm=(0.5,1),
                    size='Sensitivity',size_norm=(.5,1),sizes=(10,100))
    plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5)
    #plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5,label='Self-report')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Sensitivity', borderaxespad=0)
    plt.grid(True)
    #plt.xlabel('Average drugs per month per patient')
    #plt.ylabel('Average seizures per month per patient')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.subplot(2,2,3)
    plt.title('Clinical + Electrographic (FAR highlighted)')
    sns.color_palette("bright")
    #palette=['black','purple','cyan','orange','red','blue','green'],
    sns.scatterplot(data=p2,x='meanDrug',y='meanSz',palette='flare',hue_norm=(0,1.1),
                    style='False alarm rate', hue='False alarm rate',hue_order=flist,
                    s=100,alpha=0.8,legend=False)
    #plt.plot(xOBS,yOBS,'xr',markersize=10,alpha=0.5,label='Self-report')
    plt.grid(True)
    plt.xlabel('Average drugs per month per patient')
    plt.ylabel('Average seizures per month per patient')

    plt.subplot(2,2,4)
    plt.title('Clinical (FAR highlighted)')
    sns.scatterplot(data=p1,x='meanDrug',y='meanSz',palette='flare',hue_norm=(0,1.1),
                    style='False alarm rate',hue='False alarm rate',hue_order=flist,
                    s=100,alpha=0.8,legend='full')
    #plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5,label='(Selfreport)')
    plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5)
    
    
    
   
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='FAR', borderaxespad=0)
    plt.grid(True)
    plt.xlabel('Average drugs per month per patient')
    #plt.ylabel('Average seizures per month per patient')
    plt.ylabel('')

    
    plt.savefig(fn,dpi=300)
    plt.show()


    
def show_me_set(sens,FAR,N,clinTF,numCPUs=9,showTF=True,noFig=False,inflater=2/3,doDISCOUNT=True,findSteady=False,yrs=10):
    # define some constants
    clinic_interval = 30*3     # 3 months between clinic visits
    #yrs = 10
    L = int(yrs*12*30 / clinic_interval)
    
    # run each patient
    if numCPUs>1:
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(sim1clinic)(sens,FAR,clinic_interval,clinTF,L,inflater,doDISCOUNT,findSteady) for _ in range(N))
    else:
        temp = [ sim1clinic(sens,FAR,clinic_interval,clinTF,L,inflater,doDISCOUNT,findSteady) for _ in trange(N)]

    bigX = np.array(temp,dtype=int)
    trueCount = bigX[:,0]
    sensorCount = bigX[:,1]
    drugCountSensor = bigX[:,2]
    szFree =bigX[:,3]
    if findSteady==True:
        how_long = bigX[:,4]
    else:
        how_long = np.array([])
    
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
        # these are normalized.
        #return np.mean(szFree),trueCount/toMonths,drugCountSensor/toMonths
        if findSteady==True:
            return np.mean(szFree),trueCount,drugCountSensor,how_long
        else:
            return np.mean(szFree),trueCount,drugCountSensor

def sim1clinic(sens,FAR,clinic_interval,clinTF,L,inflater,doDISCOUNT=True,findSteady=False):
    # simulate 1 patient all the way through
    
    # constants
    #addChance = .8
    twoyears = 8        # 8 visits 3 months apart = 2 years...
    drugStrengthLO = 0.1
    drugStrengthHI = 0.2
    maxMEDS = 6.5
    #doDISCOUNT = True
    ### ADDED: chen et al indirectly tells us what percentage of patients
    #  will even TRY to get additional med added. Looking at the full table
    #  and assuming q3 months over roughly 10 years, this gives roughly 1.5%
    #  chance at any given visit of med increase.
    addChance = 0.3
    
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
    sampRATE = 1
    howmanydays = L*clinic_interval
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
                nochangeCounter=0
                if drugCount >=1:       # do nothing at all if less than 1 drug
                    drugCount -= 0.5
                    if drugCount<1:
                        drugCount = 1
                
                #since we decreased meds, szfreedom from prior drug change goes away
                szFree[i:0] = 0
                
        else:
            # things remain bad
            nochangeCounter=0
            # ready to increase something?
            # if we have no drugs on board, always do something. Otherwise SOMETIMES do something.
            if drugCount==0 or np.random.random()<addChance:
                
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
                            szFree[istart:] = 1
                        elif patternABCD==2:
                            # fluctuating sz freedom
                            imax = np.min([i + 4,L])
                            szFree[i:imax] = 1
                        else: # no szFreedom
                            szFree[i:] = 0
                            
        
        decisionList[i] = drugCount

    if findSteady:
        # figure out what the steady state was, then how long it took to get there, and return that
        SS_drugCount = np.median((decisionList[int(L*0.66):]))    # usual value in 1/3 of visits

        # I want to view 12 month intervals to smooth results a little
        windowSize = 12 * 30 / clinic_interval

        moving_median = median_filter(decisionList, size=int(windowSize))
        #windowStepSize = 1
        # create a window of ones with the same size as the window size
        #window = np.full(int(windowSize), 1.0/windowSize)
        # use np.convolve with mode='valid' to get the moving window average
        #movingAverage = np.convolve(decisionList, window, mode='valid')
        # pad the movingAverage with zeros at both ends
        #padSize = int(windowSize - windowStepSize) # calculate the padding size
        #movingAverage = np.pad(movingAverage, (padSize,0), mode='constant') # pad with zeros

        # find the locations when the moving average is close to the steady state value
        w = np.where(abs(moving_median - SS_drugCount) < 0.5 )[0]
        if len(w)==0:
            how_long = 10000
        else:
            how_long = w[0]
        debugPLOT = False
        if debugPLOT == True:
            print(f'SS = {SS_drugCount} clin_int ={clinic_interval} {w[0]}')
            plt.plot(np.arange(len(decisionList)),decisionList,label='meds')
            #plt.plot(true_clin_diary,label='true szs')
            plt.plot(np.arange(len(trueXdrugged)),trueXdrugged,label='true-drugged')
            plt.plot(np.arange(len(moving_median)),moving_median+.1,label='median filtered')
            
            plt.plot(sensorXdrugged,label='sensor-drugged')
            plt.legend()
            plt.ylim([0,7])
        vals = [ np.sum(trueXdrugged), np.sum(sensorXdrugged), np.sum(np.floor(decisionList)) , (0+np.any(szFree)), how_long]

    else:
        vals = [ np.sum(trueXdrugged), np.sum(sensorXdrugged), np.sum(np.floor(decisionList)) , (0+np.any(szFree))]

    return vals

## FOR injury case
def do_all_injury_cases(N=10000,numCPUs=9):
    xf = pd.DataFrame()
    for sens in np.linspace(0.5,1,6):
        for clinTF in [True,False]:
            x = run_injury_case(sens=sens,FAR=0,N=N,numCPUs=numCPUs,clinTF=clinTF)
            xf = pd.concat([xf,x])
    xf.to_csv('Injury_case.csv',index=False)
    print(xf)

def run_injury_case(sens,FAR,N=10000,numCPUs=9,clinTF=True):
    # see Russell-Jones et al 1989, and Neufeld et al 1999 
    # there are two different rates for different populations, so we will just get both.
    # the rates are in events per 100 patient years, so we convert to events per 1 patient day.
    #
    injury_rate_per_pt_dayA = 4.8 / (100*365)
    injury_rate_per_pt_dayB = 296 / (100*365)
    clinTFtxt = 'C' if clinTF else 'C+E'
    
    df = pd.DataFrame()
    for injuryRate in [injury_rate_per_pt_dayA,injury_rate_per_pt_dayB]:
        if numCPUs>1:
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                temp = par(delayed(simInjuries)(sens,FAR,clinTF,injuryRate) for _ in trange(N))
        else:
           temp = [ simInjuries(sens,FAR,clinTF,injuryRate) for _ in trange(N)]
        bigX = np.array(temp,dtype=float)
        noInjury = np.where(bigX[:,1]==0)[0]
        yesInjury = np.where(bigX[:,1]>0)[0]
        detectedFrac = np.mean(bigX[yesInjury,0] / bigX[yesInjury,1])
        temp = pd.DataFrame({'Seizures':[clinTFtxt],
                             'Sensitivity (%)':[100*sens],
                             'Injuries detected (%)':[int(100*detectedFrac)],
                             'Injury-free (out of 10,000)':[len(noInjury)],
                             'Total injuries':[int(np.sum(bigX[:,1]))],
                             'Simulated Injury rate per 100 patient-yrs':[int(injuryRate*100*365)],
                             'Average injuries per 100 patient-yrs':[10*np.mean(bigX[:,1])]})
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

#### SUDEP case
#

def sim_SUDEP_case(fn,reps=10000,numCPUs=9):
    xf = pd.DataFrame()
    for sens in np.linspace(0.5,1,6):
        for clinTF in [True,False]:
            x = run_SUDEP_cases(sens=sens,FAR=0,N=reps,numCPUs=numCPUs,clinTF=clinTF)
            xf = pd.concat([xf,x])
    xf.to_csv(fn,index=False)
    print(xf)
    
def run_SUDEP_cases(sens,FAR,N,numCPUs,clinTF):
    # simulate N cases and summarize
    clinTFtxt = 'C' if clinTF else 'C+E'
    if numCPUs>1:
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(sim1SUDEP)(sens,FAR,clinTF) for _ in trange(N))
    else:
        temp = [ sim1SUDEP(sens,FAR,clinTF) for _ in trange(N)]
    bigX = np.array(temp,dtype=float)

    SUDEPnum=np.sum(bigX[:,0])
    nearSUDEPnum = np.sum(bigX[:,1])
    percentPrevented = nearSUDEPnum / (nearSUDEPnum + SUDEPnum)
                        
    df = pd.DataFrame({'Seizures':[clinTFtxt],'Sensitivity':[100*sens],
                        'Prevented':[percentPrevented],
                        'SUDEP':[SUDEPnum],'nearSUDEP':[nearSUDEPnum]})
    print(df)
    return df

def sim1SUDEP(sens,FAR,clinTF):
    # simulate 1 patient 
    
    # constants
    GTC_patient_risk = 0.23
    percent_of_szs_GTC_lo = 0.09
    percent_of_szs_GTC_hi = 0.18
    inflater = 2  # this is bigger than 2/3 because otherwise it will be too small!
    yrs = 10
    howmanydays = 365*yrs
    sampRATE = 24*6 # we sample once every 10 minutes
    oneYEAR = sampRATE*365
    oneMONTH = sampRATE*30
    base_SUDEP_risk = 1.2 / (1000*oneYEAR)
    medium_SUDEP_risk = 6.1 / (1000*oneYEAR)
    high_SUDEP_risk = 18 / (1000*oneYEAR)
    
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=1)
    if clinTF==True:
        thisDiary = true_clin_diary.copy()
    else:
        thisDiary = true_e_diary.copy()

    thisSensor = add_sens_and_FAR(sensitivity=sens,FAR=FAR/sampRATE,X=thisDiary,downsampleRATE=1,inflater=inflater)
    
 
        
    # build the SUDEPrisk signal
    do_I_have_GTCs = np.random.random() < GTC_patient_risk
    if do_I_have_GTCs:
        # find all the seizures first
        inds01= np.where(thisDiary>0)
        inds = inds01[0]
        
        L = len(inds)
        # assign GTCs randomly to those
        my_GTC_rate = (np.random.random() * (percent_of_szs_GTC_hi-percent_of_szs_GTC_lo)) + percent_of_szs_GTC_lo
        coinFlips = np.random.random(L) < my_GTC_rate
        
        # compose a full GTC signal
        GTCdiary = np.zeros(howmanydays*sampRATE)
        GTCdiary[inds] = 0 + coinFlips
        
        # set up SUDEP risk based on GTCdiary
        sudepRISK = np.zeros(howmanydays*sampRATE)
        sudepRISK[0:oneYEAR] = base_SUDEP_risk
        df = pd.DataFrame({'x':GTCdiary})
        yearlyGTCcounter = np.array(df.x.rolling(oneYEAR).sum())
        sudepRISK[yearlyGTCcounter==1] = medium_SUDEP_risk
        sudepRISK[yearlyGTCcounter==2] = medium_SUDEP_risk
        sudepRISK[yearlyGTCcounter==3] = high_SUDEP_risk
    else:
        # this patient never has a GTC, so risk is base_risk
        sudepRISK = np.ones(howmanydays*sampRATE) * base_SUDEP_risk    
    
    SUDEPflag = np.random.poisson(sudepRISK)
    
    SUDEPcount = 0
    NEARsudep = 0
    if sum(SUDEPflag)>0:
        # some SUDEP may happen
        wx = np.where(SUDEPflag>0)
        sudepInds = wx[0]
        for S in sudepInds:
            # loop through each possible SUDEP
            i10 = np.where(thisDiary[S:]>0)
            i0 = i10[0]
            if len(i0)>0:
                # there is a real seizure after that SUDEP
                SUDEPind = S + i0[0]    # the index of the time of closest next seizure
                if thisSensor[SUDEPind]>0:
                    # the sensor detected this one
                    NEARsudep += 1
                else:
                    # the sensor missed this one
                    SUDEPcount += 1

    did_I_die = (SUDEPcount>0) + 0   
    return did_I_die, NEARsudep


### cluster case

def run_full_cluster_cases(fn,N=10000,numCPUs=9):
    xf = pd.DataFrame()
    for clinTF in [True,False]:    
        for sens in np.linspace(0.5,1,6):
            x = run_cluster_cases(sens=sens,FAR=0,N=N,numCPUs=numCPUs,clinTF=clinTF)
            xf = pd.concat([xf,x])
        for FAR in [.05,.1,.2,.5,1,2]:
            x = run_cluster_cases(sens=1,FAR=FAR,N=N,numCPUs=numCPUs,clinTF=clinTF)
            xf = pd.concat([xf,x],ignore_index=True)
    
    xf.to_csv(fn,index=False)
    print(xf)
    
def run_cluster_cases(sens,FAR,N,numCPUs,clinTF):
    # simulate N cases and summarize
    clinTFtxt = 'C' if clinTF else 'C+E'
    if numCPUs>1:
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(sim1clusterCase)(sens,FAR,clinTF) for _ in trange(N))
    else:
        temp = [ sim1clusterCase(sens,FAR,clinTF) for _ in trange(N)]
    bigX = np.array(temp,dtype=float)

    szCountWithout = bigX[:,0]
    szCountWith = bigX[:,1]
    theDrugs = bigX[:,2]
    szDiff=szCountWithout-szCountWith
    drugCount = np.mean(theDrugs)
    diffSz = np.mean(szDiff)
    noDrugs=np.sum(theDrugs==0)
    noSzChange = np.sum(szCountWith==szCountWithout)
    noSzChangeDrugs = np.mean(theDrugs[szCountWith==szCountWithout])
    
    diffWith = np.mean(szDiff[szDiff>0])
    drugsWith = np.mean(theDrugs[szDiff>0])
    the75 = np.percentile(szDiff,75)
    szDiffInds = np.argsort(szDiff)
    the75ind = abs(szDiff[szDiffInds]-the75).argmin()
    the75drugs = theDrugs[szDiffInds[the75ind]]
    
                    
    df = pd.DataFrame({'Type':[clinTFtxt],'sens':[sens],'FAR':[FAR],
                    'diffSz':[diffSz],'drugCount':[drugCount],
                    'noDrugs':[noDrugs],
                    'noSzChange':[noSzChange],'noSzChangeDrugs':[noSzChangeDrugs],
                    'diffWith':[diffWith],'drugsWith':[drugsWith],
                    'the75':[the75],'the75drugs':[the75drugs]})
    print(df)
    return df

def sim1clusterCase(sens,FAR,clinTF):
    inflater = 2
    yrs = 10
    
    howmanydays = 365*yrs
    sampRATE = 24*6 # we sample once every 10 minutes
    maxSamps = sampRATE*howmanydays
    sixHours = int(sampRATE / 4)
    efficacy_lo = .2
    efficacy_hi = .3
    drug_efficacy = np.random.random()*(efficacy_hi-efficacy_lo) + efficacy_lo
    # the duration of the drug will be 6 to 24 hours
    effect_duration = int(np.floor(np.random.random()*(sampRATE - sixHours) + sixHours))
    
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=1)
    if clinTF==True:
        thisDiary = true_clin_diary.copy()
    else:
        thisDiary = true_e_diary.copy()

    thisSensor = add_sens_and_FAR(sensitivity=sens,FAR=FAR/sampRATE,X=thisDiary,downsampleRATE=1,inflater=inflater)
    
    allSzCount = np.sum(thisDiary)
    
    # detect cluster (2 or more in 6 hours)
    df = pd.DataFrame({'x':thisSensor})
    temp1 = np.array(df['x'].rolling(sixHours).sum())
    windowCount = temp1.squeeze()
    # note rolling creates Nan in the first sixHours samples, so we skip those
    wx = np.where(windowCount[sixHours:]>=2)
    clusterInds = wx[0]+sixHours
    drugCount = 0
    for thisCluster in clusterInds:
        if windowCount[thisCluster]>=2:
            # meaning this sample was not yet blanked out...
            
            # first blank out the next 6 hours
            maxInd = np.min([thisCluster + sixHours,maxSamps])
            windowCount[thisCluster:maxInd] = 0
            # add to the drugCount
            drugCount +=1
            # then apply drug
            maxDrugInd = np.min([thisCluster + effect_duration,maxSamps])
            for thisSample in range(thisCluster,maxDrugInd):
                if thisDiary[thisSample]>0 and np.random.random()<drug_efficacy:
                    # there is a seizure, and the drug stopped it!
                    thisDiary[thisSample] = 0
    
    allRemainingSzCount = np.sum(thisDiary)
    
    return [allSzCount,allRemainingSzCount,drugCount]
        
def draw_cluster_summary(fn='clusterCase10k.csv',fign=''):
    p = pd.read_csv(fn)
    p1 = p[p.Seizure=='C']
    p2 = p[p.Seizure=='C+E']
    f0 = p1[p1['FAR']==0]
    s5 = f0[f0['sens']==0.5]
    xOBS = s5['drugCount'].values
    yOBS = s5['diffSz'].values

    p1['FAR'] = np.round(p1['FAR']*100) / 100
    p2['FAR'] = np.round(p2['FAR']*100) / 100
    p1 = p1.rename(columns={'FAR':'False alarm rate','sens':'Sensitivity'})
    p2 = p2.rename(columns={'FAR':'False alarm rate','sens':'Sensitivity'})
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.title('Clinical + Electrographic (sensitivity highlighted)')
    sns.color_palette("bright")
    sns.scatterplot(data=p2,x='drugCount',y='diffSz',color='k',
                    size='Sensitivity',size_norm=(.5,1),sizes=(10,100),legend=False)
    #plt.plot(xOBS,yOBS,'xr',markersize=10,alpha=0.5,label='Self-report')
    plt.grid(True)
    #plt.xlabel('Average drugs per patient')
    plt.xlabel('')
    plt.ylabel('Average seizures rescued per patient')

    plt.subplot(2,2,3)
    plt.title('Clinical + Electrographic (FAR highlighted)')
    sns.scatterplot(data=p2,x='drugCount',y='diffSz',
                    palette='viridis',hue_norm=(0,1.2),hue_order=[0,.05,.1,.2,.5,1,2],
                    hue='False alarm rate',style='False alarm rate',s=100,alpha=0.8,legend=False)
    #plt.plot(xOBS,yOBS,'xr',markersize=10,alpha=0.5,label='Self-report')
    plt.grid(True)
    plt.xlabel('Average rescue meds per patient')
    plt.ylabel('Average seizures rescued per patient')
    
    plt.subplot(2,2,2)
    plt.title('Clinical (sensitivity highlighted)')
    sns.scatterplot(data=p1,x='drugCount',y='diffSz',color='k',
                    size='Sensitivity',size_norm=(.5,1),sizes=(10,100))
    #plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5,label='Self-report')
    plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Sensitivity',borderaxespad=0)
    plt.grid(True)
    #plt.xlabel('Average drugs per patient')
    #plt.ylabel('Average seizures rescued per patient')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.subplot(2,2,4)
    plt.title('Clinical (FAR highlighted)')
    sns.scatterplot(data=p1,x='drugCount',y='diffSz',
                    palette='viridis',hue_norm=(0,1.2),hue_order=[0,.05,.1,.2,.5,1,2],
                    hue='False alarm rate',style='False alarm rate',s=100,alpha=0.8,legend='full')
    #plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5,label='Self-report')
    plt.plot(xOBS,yOBS,'xr',markersize=20,alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',title='FAR', borderaxespad=0)
    plt.grid(True)
    plt.xlabel('Average rescue meds per patient')
    #plt.ylabel('Average seizures rescued per patient')
    plt.ylabel('')
    
    if fign != '':
        plt.savefig(fign,dpi=300)
    plt.show()
    