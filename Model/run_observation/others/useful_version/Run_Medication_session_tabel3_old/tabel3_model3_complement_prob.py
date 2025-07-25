#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

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

# Get the filename of the currently running script
filename = os.path.basename(__file__)
# Remove the .py extension from the filename
model_name = os.path.splitext(filename)[0]

# session effect over Parkinsdon's disease
partcipant_group = 'HC'
  
# Main directory of the subject
readMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/'
# read collected data across all participants
behAll = pd.read_csv(f'{readMainDirec}/NoNanBehAll.csv')


# select group and condition
behAll = behAll[behAll['patient']==partcipant_group]

# number of participant
nParts = len(np.unique(behAll['sub_ID']))
# Convert participant ID to indeces
behAll['sub_ID'] = behAll['sub_ID'].replace(np.unique(behAll.sub_ID), np.arange(1, nParts +1))
# number of condition
nConds = 2
# Number of session 1 and 2
nMeds_nSes = 2
# Condition label 1: Act, label 2: Stim
behAll.block = behAll.block.replace(['Act', 'Stim'], [1, 2])
#  set the session or medication effect
if partcipant_group=='HC':
    medication_session = np.array(behAll.session).astype(int)
elif partcipant_group=='PD':
    # group label 1: PD OFF, group label 3: PD ON
    behAll['medication'] = behAll.group.replace([1, 3], [1, 2])
    medication_session = np.array(behAll.medication).astype(int)

 
# If you want to model fit or just recall ex model fit
modelFit = True
# Number of chains in MCMC procedure
n_chains = 8
# The number of iteration or samples for each chain in MCM procedure
n_samples=3000
# number of warp up samples
n_warmup = 1000
# main directory of saving
writeMainScarch = '/mnt/scratch/projects/7TPD/amin'
# The adrees name of pickle file
pickelDir = f'{writeMainScarch}/Behavioral/Tabel3/{partcipant_group}/{model_name}_{partcipant_group}.pkl'
# Check out if it does not exist
if not os.path.isdir(f'{writeMainScarch}/Behavioral/Tabel3/{partcipant_group}/'):
        os.makedirs(f'{writeMainScarch}/Behavioral/Tabel3/{partcipant_group}/') 

 
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
                'nConds':nConds,
                'condition':np.array(behAll.block).astype(int),
                'nMeds_nSes':nMeds_nSes,
                'medication_session':medication_session,
}
    # initial sampling
    initials = [] 
    for c in range(0, n_chains):
        chaininit = {
            'z_alphaAct_pos': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_alphaAct_neg': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_alphaClr_pos': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_alphaClr_neg': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_sensitivity': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'hier_alpha_sd': np.random.uniform(.01, .1),        
            'hier_sensitivity_sd': np.random.uniform(.001, .01),
            'transfer_sensitivity': np.random.uniform(.03, .07, size=(nParts, nMeds_nSes, nConds))
        }
        initials.append(chaininit)   

    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/Model/stan_models/Stan_Medication_session_tabel3/{model_name}.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
    nest_asyncio.apply()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init=initials, num_warmup=n_warmup)
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']

 
# Extracting posterior distributions for each of four main unkhown parameters
hier_weight_mu = fit["transfer_hier_weight_mu"] 
hier_alphaAct_pos_mu = fit["transfer_hier_alphaAct_pos_mu"]  
hier_alphaAct_neg_mu = fit["transfer_hier_alphaAct_neg_mu"] 
hier_alphaClr_pos_mu = fit["transfer_hier_alphaClr_pos_mu"]
hier_alphaClr_neg_mu = fit["transfer_hier_alphaClr_neg_mu"] 
hier_sensitivity_mu = fit["transfer_hier_sensitivity_mu"] 

# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(20, 8), tight_layout=True)
rows = 2
columns = 3

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(hier_weight_mu[0], kde=True, stat='density', bins=100)
sns.histplot(hier_weight_mu[1], kde=True, stat='density', bins=100)
plt.title('Hierarchical Weighting',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ w $',  fontsize=18)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)
plt.legend(['Act', 'Clr']) 
 
# Positive Learnign Rate in Action value learning
fig.add_subplot(rows, columns, 2)
sns.histplot(hier_alphaAct_pos_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaAct_pos_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaAct_pos_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaAct_pos_mu[1,1], kde=True, stat='density', bins=100)
plt.title('Hierarchical Positive Learnign Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ +\alpha_{(A)} $',  fontsize=18)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 

 
# Negative Learnign Rate in Action value learning
fig.add_subplot(rows, columns, 3)
sns.histplot(hier_alphaAct_neg_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaAct_neg_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaAct_neg_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaAct_neg_mu[1,1], kde=True, stat='density', bins=100)
plt.title('Hierarchical Negative Learnign Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ -\alpha_{(A)} $',  fontsize=18)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 

 
# Positive Learnign Rate in Color value learning
fig.add_subplot(rows, columns, 4)
sns.histplot(hier_alphaClr_pos_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaClr_pos_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaClr_pos_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaClr_pos_mu[1,1], kde=True, stat='density', bins=100)
plt.title('Hierarchical Positive Learnign Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ +\alpha_{(C)} $',  fontsize=18)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 

 
# Negative Learnign Rate in Color value learning
fig.add_subplot(rows, columns, 5)
sns.histplot(hier_alphaClr_neg_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaClr_neg_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaClr_neg_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(hier_alphaClr_neg_mu[1,1], kde=True, stat='density', bins=100)
plt.title('Hierarchical Negative Learnign Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ -\alpha_{(C)} $',  fontsize=18)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 

  
# Sensitivity
fig.add_subplot(rows, columns, 6)
sns.histplot(hier_sensitivity_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(hier_sensitivity_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(hier_sensitivity_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(hier_sensitivity_mu[1,1], kde=True, stat='density', bins=100)
plt.title('Hierarchical Sensitivity',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$\beta$',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 


# Save figure of parameter distribution 
fig.savefig(f'{writeMainScarch}/Behavioral/Tabel3/{partcipant_group}/{model_name}_{partcipant_group}.png', dpi=500)

