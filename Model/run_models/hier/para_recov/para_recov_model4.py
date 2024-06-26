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
import time

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
n_samples=5000
# Main directory of the subject
subMainDirec = '/mnt/scratch/projects/7TPD/amin/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
# number of simulation
simNumber = 5 
# Main directory of the simupated participatns
parent_dir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/'
# Directory of the especifit simulated participant
dirc = parent_dir + str(simNumber) + '/All-simulated-task-design-true-param.csv'
# Read the simulated participant
simulated_data = pd.read_csv(dirc)
simulated_data.block = simulated_data.block.replace('Stim', 'Clr')

# set of indicator to the first trial of each participant
for sub in subList:
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Clr']: # condition
                behAll_indicator = simulated_data[(simulated_data['sub_ID']==sub)&(simulated_data['block']==condition)&(simulated_data['session']==session)&(simulated_data['reverse']==reverse)]  
                simulated_data.loc[(simulated_data['sub_ID']==sub)&(simulated_data['block']==condition)&(simulated_data['session']==session)&(simulated_data['reverse']==reverse), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)

# less volatile environemnt
simulated_data = simulated_data[simulated_data['reverse']==21]
# number of participant
nParts = len(np.unique(simulated_data.sub_ID))
# participant indeces
simulated_data.sub_ID = simulated_data.sub_ID.replace(np.unique(simulated_data.sub_ID), np.arange(1, nParts +1))
# group label
nGrps = 2
simulated_data.patient = simulated_data.patient.replace(['HC', 'PD'], [1, 2])
# number of conditions
nConds = 2
simulated_data.block = simulated_data.block.replace(['Act', 'Clr'], [1, 2])
# The adrees name of pickle file
pickelDir = subMainDirec + 'Model_secondOrder/hier/para_recov/para_recov_model4_sim_' + str(simNumber)+'.pkl'
if modelFit == True: 
    """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
    # Put required data for stan model
    dataStan = {'N':simulated_data.shape[0],  
                'nParts':nParts,  
                'pushed':np.array(simulated_data.pushed).astype(int),  # should be integer
                'yellowChosen':np.array(simulated_data.yellowChosen).astype(int), # should be integer
                'winAmtPushable':np.array(simulated_data.winAmtPushable), 
                'winAmtPullable':np.array(simulated_data.winAmtPullable),
                'winAmtYellow':np.array(simulated_data.winAmtYellow), 
                'winAmtBlue':np.array(simulated_data.winAmtPullable),
                'rewarded':np.array(simulated_data.correctChoice).astype(int), # should be integer   
                'participant':np.array(simulated_data.sub_ID).astype(int),      
                'indicator':np.array(simulated_data.indicator).astype(int),   
                'nConds':nConds,
                'nGrps':nGrps,
                'condition':np.array(simulated_data.block).astype(int),
                'group': np.array(simulated_data.patient).astype(int),
                'p_push_init':.5, 
                'p_yell_init':.5}
    # initial sampling
    initials = [] 
    chaininit = {
        'alphaAct': np.random.uniform(.1, .1, size=(nParts, nConds)),
        'alphaClr': np.random.uniform(.1, .1, size=(nParts, nConds)),        
        'weightAct': np.random.uniform(.5, .5, size=(nParts, nConds)),
        'sensitivity': np.random.uniform(.01, .08, size=(nParts, nConds))
        }
    for c in range(0, n_chains):
    	initials.append(chaininit)  

    # Loading the RL Stan Model
    file_name = '/mrhome/amingk/Documents/7TPD/ActStimRL/Model/stan_models/hier/para_recov/para_recov_model4.stan' 
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
hier_alphaAct_a = fit["hier_alphaAct_a"] 
hier_alphaAct_b = fit["hier_alphaAct_b"] 
hier_alphaClr_a = fit["hier_alphaClr_a"] 
hier_alphaClr_b = fit["hier_alphaClr_b"] 
hier_weightAct_a = fit["hier_weightAct_a"] 
hier_weightAct_b = fit["hier_weightAct_b"] 
beta_ = fit["hier_sensitivity_mu"]  
# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 3
columns = 3


 # Weighting parameter distribution
fig.add_subplot(rows, columns, 1)
hier_weightAct_dist =  np.random.beta(hier_weightAct_a, hier_weightAct_b)
sns.histplot(hier_weightAct_dist[0], kde=True, stat='density')
sns.histplot(hier_weightAct_dist[1], kde=True, stat='density')
plt.legend(['Act', 'Clr'])
plt.title('Simulation ' + str(simNumber)+' -Wighting parameter distribution', fontsize=12)
plt.ylabel('Density', fontsize=8)
plt.xlabel('$w_{(A)}$', fontsize=8)
plt.subplots_adjust(wspace=10.)
plt.xlim(0,1)
  
# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(beta_[0], kde=True, stat='density')
sns.histplot(beta_[1], kde=True, stat='density')
plt.legend(['Act', 'Clr'])
plt.title('Simulation ' + str(simNumber)+' -Sensitivity parameter', fontsize=8)
plt.ylabel('Density', fontsize=8)
plt.xlabel(r'$\beta$', fontsize=8)

# Action Learning Rate distribution
fig.add_subplot(rows, columns, 3)
hier_alphaAct_dist =  np.random.beta(hier_alphaAct_a, hier_alphaAct_b)
sns.histplot(hier_alphaAct_dist[0], kde=True, stat='density')
sns.histplot(hier_alphaAct_dist[1], kde=True, stat='density')
plt.legend(['Act', 'Clr'])
plt.title('Simulation ' + str(simNumber)+' -Action Learning Rate distribution', fontsize=12)
plt.ylabel('Density', fontsize=8)
plt.xlabel(r'$ \alpha_{(A)} $', fontsize=8)
plt.subplots_adjust(wspace=10.)
plt.xlim(0,1)

# Action Learning Rate distribution
fig.add_subplot(rows, columns, 4)
hier_alphaClr_dist =  np.random.beta(hier_alphaClr_a, hier_alphaClr_b)
sns.histplot(hier_alphaClr_dist[0], kde=True, stat='density')
sns.histplot(hier_alphaClr_dist[1], kde=True, stat='density')
plt.legend(['Act', 'Clr'])
plt.title('Simulation ' + str(simNumber)+' -Color Learning Rate distribution', fontsize=12)
plt.ylabel('Density', fontsize=8)
plt.xlabel(r'$ \alpha_{(C)} $', fontsize=8)
plt.subplots_adjust(wspace=10.)
plt.xlim(0,1)

 

# Save figure of parameter distribution 
fig.savefig(subMainDirec + 'Model_secondOrder/hier/para_recov/para_recov_model4_sim_' + str(simNumber)+'.png', dpi=300)
