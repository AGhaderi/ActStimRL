#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
"""Model fit for competing Action Value Learning and Stimulus Value Learning in the cotext of Reinforcement Learning at the individual level"""
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
import nest_asyncio
import os 

# select Act or Stim to model fit seperately
cond_act_stim = 'Act'
# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']
# If you want to model fit or just recall ex model fit
modelFit = True
# Number of chains in MCMC procedure
n_chains = 5
# The number of iteration or samples for each chain in MCM procedure
n_samples=4000
# Main directory of the subject
subMainDirec = '/mnt/scratch/projects/7TPD/amin/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
# Main directory of the simupated participatns
parent_dir = '/mnt/scratch/projects/7TPD/amin/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')
   
# set of indicator to the first trial of each participant
for sub in subList:
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Stim']: # condition
                behAll_indicator = behAll[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse)]  
                behAll.loc[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)

# select stable environemnt
behAll = behAll[(behAll['block']==cond_act_stim)&(behAll['patient']=='HC')]
# number of conditions
nGrps = 2
behAll.group = behAll.group.replace([1, 3], [1, 2])
behAll.block = behAll.block.replace(['Act', 'Stim'], [1, 2])
# the list of participant
subList_PD = np.unique(behAll['sub_ID'])
print(subList_PD)
# main directory of saving
mainScarch = '/mnt/scratch/projects/7TPD/amin/'
# specific subject
for subName in subList_PD:
    # read sessions data for a specifit participant
    behAll_sub = behAll[behAll['sub_ID']==subName]  
    print('behAll subject: ', int(behAll_sub.shape[0]))
    # The adrees name of pickle file
    pickelDir = mainScarch  + 'realdata/' + subName + '/model/indv/RL_' + str(cond_act_stim)+'_session_same_lr_model5.pkl'
    if modelFit == True: 
        """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
        # Put required data for stan model
        dataStan = {'N':int(behAll_sub.shape[0]),  
                    'pushed':np.array(behAll_sub.pushed).astype(int),  # should be integer
                    'yellowChosen':np.array(behAll_sub.yellowChosen).astype(int), # should be integer
                    'winAmtPushable':np.array(behAll_sub.winAmtPushable), 
                    'winAmtPullable':np.array(behAll_sub.winAmtPullable),
                    'winAmtYellow':np.array(behAll_sub.winAmtYellow), 
                    'winAmtBlue':np.array(behAll_sub.winAmtPullable),
                    'rewarded':np.array(behAll_sub.correctChoice).astype(int), # should be integer      
                    'indicator':np.array(behAll_sub.indicator).astype(int),  
                    'nGrps':nGrps,
                    'group':np.array(behAll_sub.session).astype(int),
                    'p_push_init':.5, 
                    'p_yell_init':.5}
        # initial sampling
        initials = [] 
        for c in range(0, n_chains):
            chaininit = {
                'alpha': np.random.uniform(.6, .99),        
                'weightAct': .5,
                'sensitivity': np.random.uniform(.01, .05)
            }
            initials.append(chaininit)   

        # Loading the RL Stan Model
        file_name = '/home/amingk/Documents/7TPD/ActStimRL/Model/stan_models/indv/RL_same_lr_model5.stan' 
        file_read = open(file_name, 'r')
        stan_model = file_read.read()
        # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
        nest_asyncio.apply()
        # Building Stan Model realted to our proposed model
        posterior = stan.build(stan_model, data = dataStan)
        # Start for taking samples from parameters in the Stan Model
        fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init=initials)

        # Save Model Fit
        if not os.path.isdir(mainScarch  + 'realdata/' + subName + '/model/indv'):
                os.makedirs(mainScarch  + 'realdata/' + subName + '/model/indv')            
        utils.to_pickle(stan_fit=fit, save_path = pickelDir)
    else:
        """Loading the pickle file of model fit from the subject directory if modelFit = False"""
        loadPkl = utils.load_pickle(load_path=pickelDir)
        fit = loadPkl['fit']

    # Extracting posterior distributions for each of four main unkhown parameters
    alpha_ = fit["alpha"].flatten()
    weightAct_ = fit["weightAct"].flatten()
    beta_ = fit["sensitivity"].flatten()

    # Figure of model fit results in two column and two rows
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    rows = 2
    columns = 2

    # Weghtening
    fig.add_subplot(rows, columns, 1)
    sns.histplot(weightAct_, kde=True, stat='density') 
    plt.title(subName + ', Weighting parameter', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlabel('$w_{(A)}$', fontsize=14)
    plt.xlim(0, 1) 

    # Sensitivity
    fig.add_subplot(rows, columns, 2)
    sns.histplot(beta_, kde=True, stat='density') 
    plt.title(subName + ', Sensitivity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlabel(r'$\beta$', fontsize=14) 

    # Learning Rate
    fig.add_subplot(rows, columns, 3)
    sns.histplot(alpha_, kde=True, stat='density') 
    sns.histplot(alpha_, kde=True, stat='density')
    if cond_act_stim=='Act':
        plt.title(subName + ', Action Learning Rate', fontsize=12)
        plt.xlabel(r'$ \alpha_{(A)} $', fontsize=14)
    if cond_act_stim=='Stim':
        plt.title(subName + ', Color Learning Rate', fontsize=12)
        plt.xlabel(r'$ \alpha_{(C)} $', fontsize=14)
    plt.xlim(0, 1) 

    plt.subplots_adjust(wspace=10.)

    # Save figure of parameter distribution 
    fig.savefig(mainScarch  + 'realdata/' + subName + '/model/indv/RL_' + str(cond_act_stim)+'_session_same_lr_model5.png', dpi=300)
