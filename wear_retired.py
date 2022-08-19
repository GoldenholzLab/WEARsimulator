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
