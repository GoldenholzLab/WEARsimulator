import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import scipy
from tqdm.notebook import tqdm,trange
from weargroup_v2 import add_sens_and_FAR_onesamp,applyDrugOneSample
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
from realSim import get_mSF, simulator_base,downsample

def drawResult_from_clinic_self_report(fn,N=100000,sLIST=[1,.9,.8,.7,.6,.5,.4,.3,.2,.1],figname='Fig2-self-report.tif',doColor=True):
    #N = 100000
    #N = 1000
    #fn =f'clinicMonster_selfrep_v3with{N}.csv'

    #fn = 'clinicMonster_selfRep_1FPmonthly.csv'
    #fn = 'clinicMonster_selfrep_v2with.csv'
    plt.figure(figsize=(8,8))
    df = pd.read_csv(fn)
    df_grouped = df.groupby('sens').median()
    df_grouped['sens'] = df_grouped.index
    df_grouped['sens'] = df_grouped['sens'].astype('category')

    if doColor==True:
        pal = 'coolwarm'
        dotc = 'red'
    else:
        pal = 'Greys'
        dotc = 'black'
    plt.subplot(2,2,1)
    ax1=sns.boxenplot(data=df,x='sens',y='meanDrug',hue='sens',palette=pal)
    _lg = ax1.get_legend()
    _lg.remove()
    
    #sns.violinplot(data=df,x='sens',y='meanDrug')
    #sns.scatterplot(data=df,x='sens',y='meanDrug')
    plt.plot(np.arange(len(sLIST)),df_grouped['meanDrug'],linestyle=':',color=dotc,marker='o')
    plt.ylim(0,6)
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('Daily # meds per patient')
    plt.subplot(2,2,2)
    #ax=plt.gca()
    #ax.set(xscale="log", yscale="log")
    ax2=sns.boxenplot(data=df,x='sens',y='meanSz',hue='sens',palette=pal, legend=[np.sort(np.array(sLIST))])
    plt.legend(loc='center left', title='Sensitivity (SNR)',bbox_to_anchor=(1, 0.5))
    
    #sns.violinplot(data=df,x='sens',y='meanSz')
    plt.plot(np.arange(len(sLIST)),df_grouped['meanSz'],linestyle=':',color=dotc,marker='o')
    plt.ylim(0,15)
    plt.ylabel('Ave. sz./mo. per patient')
    plt.xlabel('')
    plt.grid(visible=True)
    plt.xticks([])
    plt.subplot(2,2,3)
    df.loc[df.how_long==10000,'how_long'] = np.NaN
    #df.how_long[df.how_long==12] = np.NaN

    ax3=sns.violinplot(data=df,x='sens',y='how_long',hue='sens',palette=pal)
    #ax = sns.boxenplot(data=df,x='sens',y='how_long')
    _lg = ax3.get_legend()
    _lg.remove()
    plt.plot(np.arange(len(sLIST)),df_grouped['how_long'],linestyle=':',color=dotc,marker='o')
    #plt.title('Sensitivity vs. how long until med stability')
    #plt.xlabel('')
    #plt.xticks([])
    plt.ylabel('Months till stable med dose')
    plt.xlabel('Sensitivity (SNR)')

    plt.subplot(2,2,4)
    # Group the dataframe by sens and compute the mean of meanSz

    ax4 = sns.scatterplot(data=df_grouped,size='sens',hue='sens',y='meanDrug',x='meanSz',
                        palette=pal,edgecolors='black')
    _lg = ax4.get_legend()
    _lg.remove()
    # Use curve_fit to fit the function to your data
    
    some_y = np.array(df_grouped.meanDrug)[::-1]
    some_x = np.array(df_grouped.meanSz)[::-1]
    #popt, pcov = curve_fit(exp_curve_func, xdata=some_x[::4], ydata=some_y[::4])
    #fit_x = np.linspace(np.min(df_grouped.meanDrug),1.9, 30)
    #model = np.poly1d(np.polyfit(some_x, some_y, 2))
    #polyline = np.linspace(np.min(df_grouped.meanSz),np.max(df_grouped.meanSz), 30)
    #plt.plot(polyline, model(polyline),'sr:')
    plt.plot(some_x,some_y,':',color=dotc,alpha=0.5)
    plt.grid(visible=True)
    #plt.plot(fit_x,exp_curve_func(fit_x, *popt), 'r:')
    #plt.legend(loc='center left', title='Sensitivity',bbox_to_anchor=(1, 0.5))
    plt.xlim(0,15)
    plt.ylim(0,4)
    plt.ylabel('Daily number of meds per patient')
    plt.xlabel('ave. sz./mo. per patient')
    # Print the result
    print(df_grouped)
    plt.savefig(figname,bbox_inches='tight',dpi=300)
    plt.show()
    
    plt.figure(figsize=(4,4))
    sns.scatterplot(data=df_grouped,size='sens',hue='sens',y='meanDrug',x='meanSz',
                        palette=pal,edgecolors='black')
    plt.legend(loc='center left', title='Sensitivity (SNR)',bbox_to_anchor=(1, 0.5))
    
    # Use curve_fit to fit the function to your data
    
    some_y = np.array(df_grouped.meanDrug)[::-1]
    some_x = np.array(df_grouped.meanSz)[::-1]
    #popt, pcov = curve_fit(exp_curve_func, xdata=some_x[::4], ydata=some_y[::4])
    #fit_x = np.linspace(np.min(df_grouped.meanDrug),1.9, 30)
    #model = np.poly1d(np.polyfit(some_x, some_y, 2))
    #polyline = np.linspace(np.min(df_grouped.meanSz),np.max(df_grouped.meanSz), 30)
    #plt.plot(polyline, model(polyline),'r:')
    #plt.plot(fit_x,exp_curve_func(fit_x, *popt), 'r:')
    plt.plot(some_x,some_y,'r:',alpha=0.5)
    plt.grid(visible=True)
    #plt.legend(loc='center left', title='Sensitivity',bbox_to_anchor=(1, 0.5))
    plt.xlim(0,15)
    plt.ylim(0,4)
    plt.ylabel('Daily number of meds per patient')
    plt.xlabel('ave. sz./mo. per patient')
    plt.show()
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    ax.set(xscale="log", yscale="log")
    ax = sns.scatterplot(data=df_grouped,size='sens',hue='sens',x='sens',y='meanSz',
                        palette=pal,edgecolors='black')
    some_y = np.array(df_grouped.meanSz)[::-1]
    some_x = np.array(df_grouped.sens)[::-1]
    #model = np.poly1d(np.polyfit(some_x, some_y, 3))
    #polyline = np.linspace(np.min(df_grouped.meanSz),13, 30)
    #ax.set_ylim(bottom=0)
    #plt.plot(np.log10(some_x),np.log10(some_y),'r:')
    plt.plot(some_x,some_y,'r:')
    plt.ylim(1,8)
    #plt.xlim(0,1)
    plt.grid(visible=True,which='both')
    plt.legend(loc='center left', title='Sensitivity (SNR)',bbox_to_anchor=(1, 0.5))
    plt.ylabel('Mean sz/month per patient')
    plt.xlabel('Sensitivity (SNR)')
    plt.show()

def do_some_self_report_sets(inflater=2/3,sLIST = [1,.9,.8],fLIST= [ 0, 0.05, 0.1],N=10000,NOFIG=False,fn='',
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
                szfree, szCounts, drugCounts, how_long = show_me_set_self_report(sens=sens,FAR=FAR,N=N,showTF=False,noFig=NOFIG,inflater=inflater,doDISCOUNT=doDISCOUNT,findSteady=findSteady,numCPUs=numCPUs,yrs=yrs)
                df = pd.concat([df,pd.DataFrame({'sens':[sens],'FAR':[FAR],'szfree':[szfree],'meanDrug':[np.median(drugCounts)/interval_count],'meanSz':[np.median(szCounts)/(interval_count*3)],'how_long':[np.median(how_long)]})])
            else:    
                szfree, szCounts, drugCounts = show_me_set_self_report(sens=sens,FAR=FAR,N=N,showTF=False,noFig=NOFIG,inflater=inflater,doDISCOUNT=doDISCOUNT,findSteady=findSteady,numCPUs=numCPUs,yrs=yrs)
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


def show_me_set_self_report(sens,FAR,N,numCPUs=9,showTF=True,noFig=False,inflater=2/3,doDISCOUNT=True,findSteady=False,yrs=10):
    # define some constants
    clinic_interval = 30*3     # 3 months between clinic visits
    #yrs = 10
    L = int(yrs*12*30 / clinic_interval)
    
    # run each patient
    if numCPUs>1:
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(sim1clinicSR)(sens,FAR,clinic_interval,L,inflater,doDISCOUNT,findSteady) for _ in range(N))
    else:
        temp = [ sim1clinicSR(sens,FAR,clinic_interval,L,inflater,doDISCOUNT,findSteady) for _ in trange(N)]

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

def sim1clinicSR(sens,FAR,clinic_interval,L,inflater,doDISCOUNT=True,findSteady=False):
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
    
    # First make 1 patient true_c
    sampRATE = 1
    howmanydays = L*clinic_interval
    true_clin_diary = make_diaries_CLIN(sampRATE,howmanydays,downsample_rate=sampRATE*clinic_interval)

    X = true_clin_diary.copy().astype('int')
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

def make_diaries_CLIN(sampRATE,howmanydays,downsample_rate=1,
                obs_sensitivity=0.5, obs_FAR=0.0):
    # modified version that does not account for electrographic seizures at all
    # does not report the self-reported version - only makes the clinically true diary
    # INPUTS:
    #  sampRATE = samples per day
    #  howmanydays = how many days to generate
    #  downsample_rate =[default 1] - downsample output by how much?
    #  obs_sensitivity [default 0.5] what fraction of clin szs are observed?
    #  obs_FAR [default 0.0] what rate of false alarms per days is used?

    # OUTPUTS:
    #  true_clin_diary - true clinical only seizures
    #  observed_clin_diary - observed clinical only seizures
    #
    #
    # USAGE:
    #true_clin_diary  =  make_diaries_CLIN(sampRATE,howmanydays)
    
    # CONSTANTS
    #obs_sensitivity = 0.5       # Elgar 2018
    #obs_FAR = 0.0               # ?? "we've got to start somewhere"
    
    # generate a monthly seizure frequency that is realistic
    mSF = get_mSF( requested_msf=-1 )
    
    # increase true SF to account for under-reporting
    mSF /= obs_sensitivity
    
    # decrease true SF to account for over-reporting
    mSF /= (1 + obs_FAR) 

    
    # generate true electrographic diary (which includes true clin szs too)
    true_clin_diary = simulator_base(sampRATE=sampRATE,number_of_days=howmanydays,defaultSeizureFreq=mSF)
    
    # downsample true diaries if requested (ie downsample_rate>1)
    if downsample_rate>1:
        true_clin_diary = downsample(true_clin_diary,downsample_rate)
        
    return true_clin_diary
