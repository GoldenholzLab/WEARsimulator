# These are functions that were tried initially but are now retired
# either due to bugs or poor optimization

def add_FAR(x,FAR):
    # INPUTS:
    #   x - the diary
    #   FAR - the rate of false alarms per sample
    # OUTPUTS:
    #   newx - the diary with extra alarms in there
    
    added_sz = np.round(np.random.randn(len(x))+FAR)
    added_sz[added_sz<0] =0
    return x + added_sz



def buildAllTrials(theData,trialCount,baseline,test,maxN,sens,FAR,numCPUs,maxPOW=0.9):
    # PRECOMPUTED MAX (using CLIN true) for 90% power is 1930 mpc, 2650 rr50
    #   for 80% it is 1460 and 2000
    
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
    xdata = theData.copy()
    #account for sensitivity
    trialDur = baseline+test
    
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
    #print(f'Number of PC nan = {np.sum(np.isnan(PC[:]))}')


    Nlist_mpc = np.arange(100,maxMPC,10)
    finalN_mpc = maxN*2
    MPC = [0,0]
    pow_list = np.zeros(len(Nlist_mpc))
    #for ni,N in enumerate(tqdm(Nlist_mpc,desc='N')):
    for ni,N in enumerate(Nlist_mpc):
    
        pow = get_pows_given_n(maxN,N,trialCount,PC,mpcTF=True,rng=rng,numCPUs=numCPUs)
        pow_list[ni] = pow
        if pow>maxPOW:
            finalN_mpc = N
            break
    
    Nlist_rr50 = np.arange(900,maxRR50,10)
    finalN_rr50 = maxN*2
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
    return finalN_mpc,finalN_rr50




def do_simple_case1_sim(figname,clinTF,numCPUs=9,REPS=10000,DRG=0.2,sensLIST = [0.5,1],FARlist = [0,10],metric_type= 'MPC'): 
    T1 = time.time()
    highestN = 1000
    
    
    if metric_type=='MPC':
        Nlist = np.arange(270,400,10)  
    else:
        Nlist = np.arange(600,900,25)
    #figname = 'oneExample.jpg'
    
    fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5,5))        
    
    #fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,5))        
    #for ic,clinTF in enumerate(tqdm([True,False],desc='Top level clinTF')):
    #    doThisIter(ax[ic],highestN,clinTF,Nlist,FARlist,sensLIST,numCPUs,REPS,DRG,metric_type)
    #clinTF=True
    doThisIter(ax,highestN,clinTF,Nlist,FARlist,sensLIST,numCPUs,REPS,DRG,metric_type)
    
       
    plt.savefig(figname,dpi=300)
    plt.show()    
    
    print(f'Time = {time.time()-T1}')
    
def doThisIter(thisax,highestN,clinTF,Nlist,FARlist,sensLIST,numCPUs,REPS,DRG,metric_type):
    # find the highest N needed from Nlist and make a heatmap of it
    L = len(FARlist)*len(sensLIST)
    k = pd.DataFrame(np.zeros((L,3)),columns=['FAR','sensitivity','N'])
    ind = 0
    finalN = highestN
    GOALp = 0.9
    REPS4 = int(REPS/4)
    maxFailed = (1-GOALp)*REPS

    for fi,FAR in enumerate(tqdm(FARlist,desc='FAR',leave=False)):
        for si,sensitivity in enumerate(tqdm(sensLIST,desc='sens',leave=False)):
            for Ni in trange(len(Nlist),desc='N',leave=False):
                N = Nlist[Ni]
                # speed optimization: if half the number of reps already
                # fails too many times, I don't need to do the second half

                p1 = get_pow_kind_sub(N,numCPUs,REPS4,DRG,sensitivity,FAR,clinTF,metric_type)
                failedREPS = (1-p1)*(REPS4)
                if failedREPS < maxFailed:
                    p2 = get_pow_kind_sub(N,numCPUs,REPS4,DRG,sensitivity,FAR,clinTF,metric_type)
                else:
                    p2 = 0
                    print('aborted after {REPS4}')
                p = (p1 + p2) / 2    
                failedREPS = (1-p)*(REPS4*2)
                if failedREPS < maxFailed:
                    p3 = get_pow_kind_sub(N,numCPUs,REPS4,DRG,sensitivity,FAR,clinTF,metric_type)
                else:
                    p3 = 0
                    print('aborted after {REPS4*2}')
                p = (p1 + p2 + p3) / 3
                failedREPS = (1-p)*(REPS4*3)
                if failedREPS < maxFailed:
                    p4 = get_pow_kind_sub(N,numCPUs,REPS4,DRG,sensitivity,FAR,clinTF,metric_type)
                else:
                    p4 = 0
                    print('aborted after {REPS4*3}')
                p = (p1 + p2 + p3 + p4) / 4
                #p = get_pow_kind_full(N,numCPUs,REPS,DRG,sensitivity,FAR,clinTF,metric_type)
                if p>GOALp:
                    finalN=N
                    break
            k.iloc[ind,:] = [FAR,sensitivity,finalN]
            ind+=1
    k['N'] = k['N'].astype('int')
    thispow = k.pivot('FAR','sensitivity','N')
    sns.heatmap(thispow, annot=True,fmt='d',linewidths=0.5, ax=thisax)
    thisax.set_title(f'Power Metric={metric_type} clinTF={clinTF}')    


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
    repSET = 1500
    minCHANGE = 0.01
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

def calculate_fisher_exact_p_value_from_responderTF(placebo_arm_responderTF,
                                   drug_arm_responderTF):

    num_placebo_arm_responders     = np.sum(placebo_arm_responderTF)
    num_drug_arm_responders        = np.sum(drug_arm_responderTF)
    num_placebo_arm_non_responders = len(placebo_arm_responderTF) - num_placebo_arm_responders
    num_drug_arm_non_responders    = len(drug_arm_responderTF)    - num_drug_arm_responders

    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])

    [_, RR50_p_value] = stats.fisher_exact(table)

    return RR50_p_value




def tryManyTrials(numCPUs,N,REPS,sensLIST,farLIST):
    PCB = 0
    DRG = 0.2
    baseline = 2
    test = 3
    halfN = int(N/2)
    
    obsPows = prepare_obs(numCPUs,N,REPS,PCB,DRG,baseline,test,halfN)
    print('doinbg the grid')
    allPows = prepare_sensfar(numCPUs,N,REPS,PCB,DRG,baseline,test,halfN,sensLIST,farLIST)
    print('done')    
    return obsPows,allPows

def prepare_sensfar(numCPUs,N,REPS,PCB,DRG,baseline,test,halfN,sensLIST,farLIST):
    MandR = np.zeros((len(sensLIST),len(farLIST),REPS,4))
    for Si,S in enumerate(sensLIST):
        for Fi,F in enumerate(farLIST):
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                temp = par(delayed(prepare_sensfar_sub)(N,PCB,DRG,baseline,test,halfN,S,F) for _ in trange(REPS,desc='reps'))
            MandR[Si,Fi,:,:] = np.array(temp,dtype=int)
    allPows = np.mean(MandR,axis=2)
    return allPows

def prepare_obs(numCPUs,N,REPS,PCB,DRG,baseline,test,halfN):
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(prepare_obs_sub)(N,PCB,DRG,baseline,test,halfN) for _ in trange(REPS,desc='obs'))
    MandR = np.array(temp,dtype=int)
    obsPows = np.mean(MandR,axis=0)
    return obsPows

def prepare_obs_sub(N,PCB,DRG,baseline,test,halfN):
    obs_sens = 0.5
    obs_FAR = 0
    temp = make_some_pts(sensitivity=obs_sens,FAR=obs_FAR,Ne=0,Nc = N,PCB=PCB,DRG=DRG,baseline=baseline,test=test)
    obs = getPC(temp[1,:,:],baseline,test)
    pMPCo = calculate_MPC_p_value(obs[:halfN],obs[halfN:]) < 0.05
    pRRo = calculate_fisher_exact_p_value(obs[:halfN],obs[halfN:]) < 0.05
    return np.array([pMPCo,pRRo]).astype('int')

def prepare_sensfar_sub(N,PCB,DRG,baseline,test,halfN,S,F):
    temp = make_some_pts(sensitivity=S,FAR=F,Ne=N,Nc = N,PCB=PCB,DRG=DRG,baseline=baseline,test=test)
    PCe = getPC(temp[0,:,:],baseline,test)
    PCc = getPC(temp[0,:,:],baseline,test)
    
    pMPCe = calculate_MPC_p_value(PCe[:halfN],PCe[halfN:]) < 0.05
    pMPCc = calculate_MPC_p_value(PCc[:halfN],PCc[halfN:]) < 0.05
    pRRe = calculate_fisher_exact_p_value(PCe[:halfN],PCe[halfN:]) < 0.05
    pRRc = calculate_fisher_exact_p_value(PCc[:halfN],PCc[halfN:]) < 0.05
    bigX = np.array([pMPCe,pRRe,pMPCc,pRRc])
    return bigX.astype('int')
    
    
def make_some_pts(sensitivity,FAR,Ne,Nc,PCB,DRG,baseline,test):
    sampRATE = 28
    minSz = 4
    trialDur = baseline + test
    manyPts = np.zeros((2,np.max([Ne,Nc]),trialDur))
    counters = np.array([0,0])
    while (counters[0]<Ne or counters[1]<Nc):
        X = make_one_multi(trialDur)
        Xs = applyDrug(efficacy=sensitivity,x=X,baseline=0)
        thisFAR = np.max([0,FAR + FAR*np.random.randn()])
        Xsf = Xs + sampRATE*thisFAR
        e = Xsf[:trialDur]
        c = Xsf[trialDur:]
        eT = isQualified(e,minSz,baseline)
        cT = isQualified(c,minSz,baseline) 
        if eT and counters[0]<Ne:
            temp = applyDrug(efficacy=PCB,x=e,baseline=baseline)
            if counters[0]>=(Ne/2):
                temp = applyDrug(efficacy=DRG,x=temp,baseline=baseline)
            manyPts[0,counters[0],:] = temp
            counters[0] += 1
        if cT and counters[1]<Nc:
            temp = applyDrug(efficacy=PCB,x=c,baseline=baseline)
            if counters[0]>=(Nc/2):
                temp = applyDrug(efficacy=DRG,x=temp,baseline=baseline)
            manyPts[1,counters[1],:] = temp
            counters[1] += 1

    return manyPts
    
def make_one_multi(trialDur):
    howmanydays = 28*trialDur
    trialSet = np.zeros((2,trialDur))
    e,c = make_multi_diaries(sampRATE=1,howmanydays=howmanydays,makeOBS=False,downsample_rate=28)
    return np.concatenate([e,c])
        
def isQualified(x,minSz,baseline):
    return np.mean(x[:baseline]) >= minSz



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
    
    
def show_me_one(sens,FAR,N,yrs=10,numCPUs=9):
    showEACH=False
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
    
    #plt.plot(t,np.median(mypt[:,0,:],axis=0),'-b',label='e')
    ##plt.subplot(3,1,2)
    Xe = np.zeros((N,40)).astype('int')
    Xc = np.zeros((N,40)).astype('int')
    Xe_sim = np.zeros((N,40))
    Xc_sim = np.zeros((N,40))
    trueX_sim = np.zeros((N,40))
    decisionList_clin = np.zeros((N,40))
    decisionList_e = np.zeros((N,40))
    decisionList_true = np.zeros((N,40))
    szfree_t = np.zeros((N,40))
    szfree_e = np.zeros((N,40))
    szfree_c = np.zeros((N,40))
    howManyC = np.zeros(N)
    howManyE = np.zeros(N)
    for i in trange(N):
        Xe[i,:] = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[i,0,:],downsampleRATE=clinic_interval,inflater=2/3)
        Xc[i,:] = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[i,1,:],downsampleRATE=clinic_interval,inflater=2/3)

        decisionList_clin[i,:],decisionList_e[i,:],decisionList_true[i,:],trueX_sim[i,:],Xe_sim[i,:],Xc_sim[i,:],szfree_t[i,:],szfree_e[i,:],szfree_c[i,:],howManyE[i],howManyC[i] = simulate_1pt_in_clinic(mypt[i,1,:],Xe[i,:],Xc[i,:],FAR,sens,clinic_interval)
           
    #plt.plot(t,np.median(Xe,axis=0),':b',label='e-device')
    #plt.plot(t,np.median(Xc,axis=0),':r',label='clin-device')
    
 
    
    #plt.subplot(3,1,3)
    #plt.plot(t,np.median(Xe_sim,axis=0),'--b',label='e-device-meds')
    #plt.plot(t,np.median(Xc_sim,axis=0),'--r',label='clin-device-meds')
    #plt.subplots(3,1,sharex=True,sharey=False,figsize=(4,10))
    #plt.subplot(3,1,1)
    #plt.plot(t,np.mean(trueX_sim,axis=0),'--k',label='true')
    #plt.plot(t,np.mean(Xe_sim,axis=0),'--b',label='e-device-meds')
    #plt.plot(t,np.mean(Xc_sim,axis=0),'--r',label='clin-device-meds')
    
    #plt.title('ave sz rates')
    theBINS = [0,.5,.75,1.25,1.5,2]
    hC,b = np.histogram(howManyC,bins=theBINS)
    hE,b = np.histogram(howManyE,bins=theBINS)
    print(hC)
    #plt.plot(theBINS[:-1],hC,label='C',alpha=0.5)
    #plt.plot(theBINS[:-1],hE,label='E',alpha=0.5)
    
    #plt.bar(x=np.arange(len(theBINS)),height=hC,label='C',alpha=0.5)
    #plt.bar(x=np.arange(len(theBINS)),height=hE,label='E',alpha=0.5)
    #plt.hist(howManyC,bins=theBINS,label='clin',alpha=0.5,density=True)
    #plt.hist(howManyE,bins=theBINS,label='e',alpha=0.5,density=True)
    #plt.title('Ratio of seizures')
    #plt.legend()
    #plt.show()

    plt.subplots(2,1,sharex=True,sharey=False,figsize=(4,8))
    plt.subplot(2,1,1)    
    plt.plot(t,100*np.mean(szfree_t,axis=0),label='true')
    plt.plot(t,100*np.mean(szfree_e,axis=0),label='e')
    plt.plot(t,100*np.mean(szfree_c,axis=0),label='c')
    plt.ylim([0,100])
    plt.grid(True)
    plt.legend()
    plt.title('Sz free %')
    #plt.show()
    
    #plt.plot(t,np.median(decisionList_true,axis=0),label='true')
    #plt.plot(t,np.median(decisionList_e,axis=0),label='e')
    #plt.plot(t,np.median(decisionList_clin,axis=0),label='clin')
     
    plt.subplot(2,1,2)   
    if 1:
        D = np.zeros((3,40))
        for ti in range(40):
            thisSlice = szfree_t[:,ti]==0
            if np.sum(thisSlice)>0:
                D[0,ti] = np.mean(decisionList_true[thisSlice,ti])
            thisSlice = szfree_e[:,ti]==0
            if np.sum(thisSlice)>0:
                D[1,ti] = np.mean(decisionList_e[thisSlice,ti])
            thisSlice = szfree_c[:,ti]==0
            if np.sum(thisSlice)>0:
                D[2,ti] = np.mean(decisionList_clin[thisSlice,ti])
            
        plt.plot(t,D[0,:],'r',label='true')
        plt.plot(t,D[1,:],'b',label='e')
        plt.plot(t,D[2,:],'k',label='clin')
    #plt.plot(t,np.mean(decisionList_true,axis=0),label='true')
    #plt.plot(t,np.mean(decisionList_e,axis=0),label='e')
    #plt.plot(t,np.mean(decisionList_clin,axis=0),label='c')

    #plt.boxplot(decisionList_true)
    plt.legend()
    plt.title('med counts')
    plt.show()

    if 0:
        plt.plot(t,np.mean(mypt[:,1,:],axis=0),'-r',label='clin')     
        plt.plot(t,np.mean(Xe,axis=0),':b',label='e-device')
        plt.plot(t,np.mean(Xc,axis=0),':r',label='clin-device')
        plt.title('device applied')
        plt.legend()
        plt.show()
        

def do_one_pt_clinic_rates(mypt,clinic_interval,sens,FAR):
    
    Xe = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[0,:],downsampleRATE=clinic_interval,inflater=2/3)
    Xc = add_sens_and_FAR(sensitivity=sens,FAR=FAR,X=mypt[1,:],downsampleRATE=clinic_interval,inflater=2/3)
    
    decisionList_clin,decisionList_e,decisionList_true,_,_,_,_,_,_ = simulate_1pt_in_clinic(mypt[1,:],Xe,Xc,FAR,sens,clinic_interval)
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


def simulate_1pt_in_clinic(myptT,Xe,Xc,FAR,sens,clinic_interval):    
    L = len(myptT)
    decisionList_true, newMypt,szFree_true,_ = do_decisions(myptT,myptT,L,FAR,sens,clinic_interval)
    decisionList_e, newXe,szFree_e,trueXe = do_decisions(myptT,Xe,L,FAR,sens,clinic_interval)
    decisionList_c, newXc, szFree_c,trueXc = do_decisions(myptT,Xc,L,FAR,sens,clinic_interval)
    
    numS = np.sum(newMypt)
    howManyC = np.divide(np.sum(trueXc),numS)
    howManyE = np.divide(np.sum(trueXe),numS)
    return decisionList_c,decisionList_e,decisionList_true,newMypt,newXe,newXc,szFree_true,szFree_e,szFree_c,howManyE,howManyC

def do_decisions(trueX,myX,L,FAR,sens,clinic_interval):
    X = myX.copy()
    Xt = trueX.copy()
    
    addChance = .8
    twoyears = 8        # 8 visits 3 months apart = 2 years...
    drugStrength = 0.2
    decisionList = np.zeros(L).astype('int')
    strengthList = np.zeros(L)
    szFree = np.zeros(L).astype('int')
    nochangeCounter = 0
    maxMEDS = 6

    ### Model of sz freedom (Chen et al 2018) (note we use percentage of total cohort % values)
    ### also Brodie et al 2012
    r = np.random.random()
    patternCutoffs = np.cumsum([0.37,0.22,0.16,0.25])
    if r<patternCutoffs[0]:
        patternABCD = 0
    elif r<patternCutoffs[1]:
        patternABCD = 1
    elif r<patternCutoffs[2]:
        patternABCD = 2
    else:
        patternABCD = 3
        
    successLIST = [0,.46,.28,.24,.15,0.14,0.14]
    #successLIST = [0,.72,.18,.07,.02,0.01,0.01]
    
    szFreeChances = np.less(np.random.random(maxMEDS+1),successLIST)
    # the first 3 months will have a decision of do nothing, so start on the second visit.
    for i in range(1,L):
        # first, apply the previous decision to this sample
        if szFree[i] == 1:
            X[i] = 0
            Xt[i]=0
        else:
            if decisionList[i-1]>0:
                temp = trueX[i]
                for drugNum in range(decisionList[i-1]):    
                    if drugNum==(decisionList[i-1]-1):
                        eff = strengthList[i]
                    else:
                        eff = drugStrength
                    temp = applyDrugOneSample(samp=temp,efficacy=eff)
                X[i]=temp
                Xt[i]=temp

                X[i] = add_sens_and_FAR_onesamp(sensitivity=sens,FAR=FAR,X=X[i],downsampleRATE=clinic_interval,inflater=2/3)
        # now make this clinic's decision based on the result        
        if X[i]<= (0.5 * X[i-1]):
            # no change condition
            decisionList[i] = decisionList[i-1]
            strengthList[i] = strengthList[i-1]
            # check if 2 yrs sz free
            nochangeCounter += 1
            if nochangeCounter==twoyears:
                # nothing bad in 2 years, decrease med
                if strengthList[i]==0.2:
                    strengthList[i] = 0.1
                else:
                    decisionList[i] = np.max([ (decisionList[i-1] - 1) ,0])
                    strengthList[i] = 0.2
                nochangeCounter = 0
        else:
            # add med condition... 
            # probabilistically add a med
            coinFlip = np.random.random()<addChance
            if coinFlip==True:
                # now actually add a med
                nochangeCounter = 0
                if strengthList[i-1]==0.1:
                    # higher strength only
                    strengthList[i] = 0.2
                    decisionList[i] = decisionList[i-1]
                else:
                    # new med altogether
                    decisionList[i] = decisionList[i-1]+1        
                    decisionList[i] = np.min([decisionList[i],maxMEDS])
                    strengthList[i] = 0.1
                    
                    if szFreeChances[decisionList[i]]==True:
                        if patternABCD==0: 
                            # sustained sz free
                            szFree[i:] = 1
                        elif patternABCD==1:
                            # delayed sz free sustained
                            delayT = np.random.randint(4)
                            istart = np.min([i+delayT,L-1])
                            szFree[istart:]
                        elif patternABCD==2:
                            # fluctuating sz freedom
                            imax = np.min([i + 4,L])
                            szFree[i:imax] = 1
                        # otherwise you are not allowed sz freedom, sorry!
            else:
                decisionList[i] = decisionList[i-1]
                strengthList[i] = strengthList[i-1]
                        
    return decisionList,X,szFree,Xt
                
def build_mypt(yrs,clinic_interval):
    sampRATE = 6
    howmanydays = yrs*30*12
    true_e_diary, true_clin_diary = make_multi_diaries(sampRATE,howmanydays,makeOBS=False,downsample_rate=sampRATE*clinic_interval)

    return np.concatenate([[true_e_diary], [true_clin_diary]])