#!/mrhome/amingk/anaconda3/bin/python

import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
sys.path.append('..')
from madule import utils
import nest_asyncio

# If you want to model fit or just recall ex model fit
modelFit = True
# Number of chains in MCMC procedure
nChains = 2
# The number of iteration or samples for each chain in MCM procedure
nSamples=2000 

# Main directory
mainDir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/fit_originalfMRIbehFiles/hierarchical/'
# The adrees name of pickle file
pickelDir = mainDir + 'RL_hier_3.pkl'

if modelFit == True: 
    """Fitting data to model and then save as pickle file in the subject directory if modelFit = True""" 
    # read all data
    dataAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/First-level-analysis/behAll.csv')
    # data from group label 2 particiapnt
    dataGroup2 = dataAll.loc[dataAll['group']==2]   

    nCond = 2 
    nSes = 2 
    # number of participant
    nParts = len(np.unique(dataGroup2.sub_ID))
    # participant indeces
    participant = dataGroup2.sub_ID.replace(np.unique(dataGroup2.sub_ID),
                          np.arange(1, nParts +1, 1))
    # condition indeces
    condition = dataGroup2.block.replace('Act',1).replace('Stim',2)

    # Put required data for stan model
    dataStan = {'N':int(dataGroup2.shape[0]),  
                'nParts': nParts,
                'nCond':2, 
                'nSes':2, 
                'pushed':np.array(dataGroup2.pushed).astype(int),  
                'yellowChosen':np.array(dataGroup2.yellowChosen).astype(int), 
                'winAmtPushable':np.array(dataGroup2.winAmtPushable).astype(int), 
                'winAmtYellow':np.array(dataGroup2.winAmtYellow).astype(int), 
                'rewarded':np.array(dataGroup2.correctChoice).astype(int),  
                'p_push_init':.5, 
                'p_yell_init':.5,        
                'participant':np.array(participant).astype(int),      
                'session':np.array(dataGroup2.session).astype(int),
                'condition':np.array(condition).astype(int)}

    # initial sampling
    initials = [] 
    for c in range(0, nChains):
        chaininit = {
            'alphaAct_sd': np.random.uniform(.05, .2),
            'alphaClr_sd': np.random.uniform(.05, .2),        
            'weightAct_sd': np.random.uniform(.05, .2),
            'sensitivity_sd': np.random.uniform(.05, .2),
            'alphaAct_hier': np.random.uniform(.3, .7, size=(nSes, nCond)),
            'alphaClr_hier': np.random.uniform(.3, .7, size=(nSes, nCond)),
            'weightAct_hier': np.random.uniform(.3, .7, size=(nSes, nCond)),        
            'sensitivity_hier': np.random.uniform(.1, .1, size=(nSes)),
            'alphaAct': np.random.uniform(.1, .9, size=(nParts, nSes, nCond)),       
            'alphaClr': np.random.uniform(.1, .9, size=(nParts, nSes, nCond)),
            'weightAct': np.random.uniform(.1, .9, size=(nParts, nSes, nCond)),   
            'sensitivity': np.random.uniform(.1, .2, size=(nParts, nSes))
        }
        initials.append(chaininit)   

    # Loading the RL Stan Model
    file_name = '../stan_models/RL_hier_3.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
    nest_asyncio.apply()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=nChains, num_samples=nSamples, init = initials)
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']
    
# Extracting posterior distributions for each of four main unkhown parameters
alphaAct_ = fit["alphaAct_hier"] 
alphaClr_ = fit["alphaClr_hier"] 
weightAct_ = fit["weightAct_hier"] 
beta_ = fit["sensitivity_hier"] 

# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(weightAct_[0, 0], kde=True, stat='density')
sns.histplot(weightAct_[0, 1], kde=True, stat='density')
sns.histplot(weightAct_[1, 0], kde=True, stat='density')
sns.histplot(weightAct_[1, 1], kde=True, stat='density')
plt.title('Hierarchical Weightening', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel('$w_{(A)}$', fontsize=14)
plt.legend(['Ses 1, Act', 'Ses 1, Clr', 'Ses 2, Act', 'Ses 2, Clr'], fontsize=8)

# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(beta_[0], kde=True, stat='density')
sns.histplot(beta_[1], kde=True, stat='density')
plt.title('Hierarchical Sensitivity', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$\beta$', fontsize=14)
plt.legend(['Ses 1', 'Ses 2'], fontsize=8)

# Action Learning Rate
fig.add_subplot(rows, columns, 3)
sns.histplot(alphaAct_[0, 0], kde=True, stat='density')
sns.histplot(alphaAct_[0, 1], kde=True, stat='density')
sns.histplot(alphaAct_[1, 0], kde=True, stat='density')
sns.histplot(alphaAct_[1, 1], kde=True, stat='density')
plt.title('Hierarchical Action Learning Rate', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$ \alpha_{(A)} $', fontsize=14)
plt.legend(['Ses 1, Act', 'Ses 1, Clr', 'Ses 2, Act', 'Ses 2, Clr'], fontsize=8)

# Color Learning Rate
fig.add_subplot(rows, columns, 4)
sns.histplot(alphaClr_[0, 0], kde=True, stat='density')
sns.histplot(alphaClr_[0, 1], kde=True, stat='density')
sns.histplot(alphaClr_[1, 0], kde=True, stat='density')
sns.histplot(alphaClr_[1, 1], kde=True, stat='density')
plt.title('Hierarchical Color Learning Rate', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel(r'$ \alpha_{(C)} $', fontsize=14)
plt.legend(['Ses 1, Act', 'Ses 1, Clr', 'Ses 2, Act', 'Ses 2, Clr'], fontsize=8)

plt.subplots_adjust(wspace=10.)

fig.savefig(mainDir +'RL_hier_3.png', dpi=300)
