#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
"""Model 1 provide medication effect (OFF vs ON) in Parkinson's disease in both Action and Color values leanring condition.
alpha : Learning rate for Action/Color value learning
weightAct : Weighting pratameter showing Relative contribution of Action Value Learning verus Color Value Learning
beta : Sensitivity parameter
It assumes Medication anc conidtion can change all latent parameters, therfore we will have the follwong number of parameters
alpha[2,2] two Medication effect (OFF vs ON), two Conditions [Act, Clr]
weightAct[2,2]  two Medication effect (OFF vs ON), two Conditions [Act, Clr]
beta[2,2]  two Medication effect (OFF vs ON), two Conditions [Act, Clr]
"""

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

# Medication effect over Parkinsdon's disease
partcipant_group = 'PD'
# Get the filename of the currently running script
filename = os.path.basename(__file__)
# Remove the .py extension from the filename
model_name = os.path.splitext(filename)[0]
 
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
n_chains = 4
# The number of iteration or samples for each chain in MCM procedure
n_samples=4000
# Main directory of the subject
subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
# read collected data across all participants
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')

# set of indicator to the first trial of each participant
for sub in subList:
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Stim']: # condition
                behAll_indicator = behAll[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse)]  
                behAll.loc[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)

# select PD group
behAll = behAll[behAll['patient']==partcipant_group]

# Number of Medication effect (OFF vs ON) 
nMeds = 2
# Number of conditions (Action vs Color)
nConds = 2
# group label 1: PD OFF, group label 3: PD ON
behAll['medication'] = behAll.group.replace([1, 3], [1, 2])
# number of participant
nParts = len(np.unique(behAll.sub_ID))
# participant indeces
behAll.sub_ID = behAll.sub_ID.replace(np.unique(behAll.sub_ID), np.arange(1, nParts +1))
# main directory of saving
mainScarch = '/mnt/scratch/projects/7TPD/amin'
# The adrees name of pickle file
pickelDir = f'{mainScarch}/realdata/hier/{partcipant_group}/{model_name}.pkl'
if modelFit == True: 
    """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
    # Put required data for stan model
    dataStan = {'N':behAll.shape[0],  
                'nParts':nParts,  
                'pushed':np.array(behAll.pushed).astype(int),  # should be integer
                'yellowChosen':np.array(behAll.yellowChosen).astype(int), # should be integer
                'winAmtPushable':np.array(behAll.winAmtPushable), 
                'winAmtPullable':np.array(behAll.winAmtPullable),
                'winAmtYellow':np.array(behAll.winAmtYellow), 
                'winAmtBlue':np.array(behAll.winAmtPullable),
                'rewarded':np.array(behAll.correctChoice).astype(int), # should be integer   
                'participant':np.array(behAll.sub_ID).astype(int),      
                'indicator':np.array(behAll.indicator).astype(int),  
                'nMeds_nSes':nMeds,
                'nConds':nConds,
                'medication_session':np.array(behAll.medication).astype(int),
                'condition':np.array(behAll.block).astype(int),
                'p_push_init':.5, 
                'p_yell_init':.5}
    # initial sampling
    initials = [] 
    for c in range(0, n_chains):
        chaininit = {
            'z_alpha': np.random.uniform(-1, 1, size=(nParts, nMeds, nConds)),
            'z_weight': np.random.uniform(-1, 1, size=(nParts, nMeds, nConds)),
            'z_sensitivity': np.random.uniform(-1, 1, size=(nParts, nMeds, nConds)),
            'hier_alpha_sd': np.random.uniform(.01, .1),        
            'hier_weight_sd': np.random.uniform(.01, .1),
            'hier_sensitivity_sd': np.random.uniform(.01, .1),
        }
        initials.append(chaininit)   

    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/Model/stan_models/hier/{model_name}.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
    nest_asyncio.apply()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init=initials)
    # Save Model Fit
    if not os.path.isdir(mainScarch  + '/realdata/hier/PD-HC/diagnosis/'):
            os.makedirs(mainScarch  + '/realdata/hier/PD-HC/diagnosis/') 
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']

# Extracting posterior distributions for each of four main unkhown parameters
alpha = fit["transfer_hier_alpha_mu"] 
weight = fit["transfer_hier_weight_mu"] 
beta = fit["transfer_hier_sensitivity_mu"]
# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(weight[0,0], kde=True, stat='density', bins=100)
sns.histplot(weight[0,1], kde=True, stat='density', bins=100)
sns.histplot(weight[1,0], kde=True, stat='density', bins=100)
sns.histplot(weight[1,1], kde=True, stat='density', bins=100)
plt.title('Weighting parameter', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel('$w_{(A)}$', fontsize=14)
plt.xlim(0, 1)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr']) 

# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(beta[0,0], kde=True, stat='density', bins=100)
sns.histplot(beta[0,1], kde=True, stat='density', bins=100)
sns.histplot(beta[1,0], kde=True, stat='density', bins=100)
sns.histplot(beta[1,1], kde=True, stat='density', bins=100)
plt.title('Sensitivity', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$\beta$', fontsize=14)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr']) 

# Action Learning Rate
fig.add_subplot(rows, columns, 3)
sns.histplot(alpha[0,0], kde=True, stat='density', bins=100)
sns.histplot(alpha[0,1], kde=True, stat='density', bins=100)
sns.histplot(alpha[1,0], kde=True, stat='density', bins=100)
sns.histplot(alpha[1,1], kde=True, stat='density', bins=100)
plt.title('Learning Rate', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$ \alpha$', fontsize=14)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr']) 

# Save figure of parameter distribution 
fig.savefig(f'{mainScarch}/realdata/hier/{partcipant_group}/{model_name}.png', dpi=300)

# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2
 