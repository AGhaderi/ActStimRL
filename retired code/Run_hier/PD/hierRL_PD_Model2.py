#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
"""Model fit for competing Action Value Learning and Stimulus Value Learning in the cotext of Reinforcement Learning at the hierarchical level.
It is based on Group 2"""

import numpy as np #
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
import nest_asyncio

# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']

# If you want to model fit or just recall ex model fit
modelFit = False
# Number of chains in MCMC procedure
n_chains = 5
# The number of iteration or samples for each chain in MCM procedure
n_samples=5000
# Main directory of the subject
subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')
behAll.block = behAll.block.replace('Stim', 'Clr')

# set of indicator to the first trial of each participant
for sub in subList:
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Clr']: # condition
                behAll_indicator = behAll[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse)]  
                behAll.loc[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)

# select the healthy control data
behAll_group = behAll[behAll['patient']=='PD']
# number of participant
nParts = len(np.unique(behAll_group.sub_ID))
# participant indeces
behAll_group.sub_ID = behAll_group.sub_ID.replace(np.unique(behAll_group.sub_ID), np.arange(1, nParts +1))
# number of medications
nMeds = 2
behAll_group.group = behAll_group.group.replace([1, 3], [1, 2])
# number of conditions
nConds = 2
behAll_group.block = behAll_group.block.replace(['Act', 'Clr'], [1, 2])
# read all conditions data for a specifit participant
# The adrees name of pickle file
pickelDir = subMainDirec + 'Model_secondOrder/hier/group13/pool/hierRL_PD_Model2.pkl'
if modelFit == True: 
    """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
    # Put required data for stan model
    dataStan = {'N':behAll_group.shape[0],  
                'nParts':nParts,  
                'pushed':np.array(behAll_group.pushed).astype(int),  # should be integer
                'yellowChosen':np.array(behAll_group.yellowChosen).astype(int), # should be integer
                'winAmtPushable':np.array(behAll_group.winAmtPushable), 
                'winAmtPullable':np.array(behAll_group.winAmtPullable),
                'winAmtYellow':np.array(behAll_group.winAmtYellow), 
                'winAmtBlue':np.array(behAll_group.winAmtPullable),
                'rewarded':np.array(behAll_group.correctChoice).astype(int), # should be integer   
                'participant':np.array(behAll_group.sub_ID).astype(int),      
                'indicator':np.array(behAll_group.indicator).astype(int),   
                'nConds':nConds,
                'condition':np.array(behAll_group.block).astype(int),
                'p_push_init':.5, 
                'p_yell_init':.5}
    # initial sampling
    initials = [] 
    for c in range(0, n_chains):
        chaininit = {
            'z_alphaAct': np.random.uniform(-1, 1, size=(nParts)),
            'z_alphClr': np.random.uniform(-1, 1, size=(nParts)),        
            'z_weightAct': np.random.uniform(-1, 1, size=(nParts, nConds)),
            'z_sensitivity': np.random.uniform(-1, 1, size=(nParts)),
            'hier_alphaAct_sd': np.random.uniform(.01, .1),
            'hier_alphaClr_sd': np.random.uniform(.01, .1),        
            'hier_weightAct_sd': np.random.uniform(.01, .1),
            'hier_sensitivity_sd': np.random.uniform(.01, .1),
        }
        initials.append(chaininit)   

    # Loading the RL Stan Model
    file_name = '/mrhome/amingk/Documents/7TPD/ActStimRL/Model/stan_models/hier/PD/hierRL_PD_Model2.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
    nest_asyncio.apply()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init=initials)
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']

# Extracting posterior distributions for each of four main unkhown parameters
alphaAct_ = fit["transfer_hier_alphaAct_mu"].flatten() 
alphaClr_ = fit["transfer_hier_alphaClr_mu"].flatten() 
weightAct_ = fit["transfer_hier_weightAct_mu"] 
beta_ = fit["transfer_hier_sensitivity_mu"].flatten()  
# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(weightAct_[0], kde=True, stat='density', bins=100)
sns.histplot(weightAct_[1], kde=True, stat='density', bins=100)
plt.title('Weightening parameter in PD', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel('$w_{(A)}$', fontsize=14)
plt.legend(['Act', 'Clr'])
plt.xlim(0, 1)

# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(beta_, kde=True, stat='density', bins=100)
plt.title('Sensitivity parameter in PD', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$\beta$', fontsize=14)

# Action Learning Rate
fig.add_subplot(rows, columns, 3)
sns.histplot(alphaAct_, kde=True, stat='density', bins=100)
plt.title('Action Learning Rate parameter in PD', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$ \alpha_{(A)} $', fontsize=14)
plt.xlim(0, 1)

# Color Learning Rate
fig.add_subplot(rows, columns, 4)
sns.histplot(alphaClr_, kde=True, stat='density', bins=100)
plt.title('Color Learning Rate parameter in PD', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$ \alpha_{(C)} $', fontsize=14)
plt.xlim(0, 1)
plt.subplots_adjust(wspace=10.)

# Save figure of parameter distribution 
fig.savefig(subMainDirec + 'Model_secondOrder/hier/group13/pool/hierRL_PD_Model2.png', dpi=300)
