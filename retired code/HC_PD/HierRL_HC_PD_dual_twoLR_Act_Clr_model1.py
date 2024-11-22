#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
"""Model 1 provide session effect (OFF vs ON) in Parkinson's disease in both Action and Color values leanring condition.
alphaAct_pos : Positive Learning rate for Action Value Learning
alphaAct_pos : Negative Learning rate for Action Value Learning
alphaClr_pos : Positive Learning rate for Color value learning
alphaClr_neg : Negative Learning rate for Color value learning
weight : Weighting pratameter showing Relative contribution of Action Value Learning verus Color Value Learning
beta : Sensitivity parameter
It assumes session anc conidtion can change all latent parameters, therfore we will have the follwong number of parameters
alphaAct_pos[2,2] two session effect (OFF vs ON), two Conditions [Act, Clr]
alphaAct_pos[2,2] two session effect (OFF vs ON), two Conditions [Act, Clr]
alphaClr_pos[2,2]  two session effect (OFF vs ON), two Conditions [Act, Clr]
alphaClr_neg[2,2]  two session effect (OFF vs ON), two Conditions [Act, Clr]
weight[2,2]  two session effect (OFF vs ON), two Conditions [Act, Clr]
beta[2,2]  two session effect (OFF vs ON), two Conditions [Act, Clr]
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
modelFit = True
# Number of chains in MCMC procedure
n_chains = 1
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


"""select HC group"""
behAll_HC = behAll[behAll['patient']=='HC']
# Number of session 1 and 2
nSes_HC = 2
# Number of conditions (Action vs Color)
nConds_HC = 2
# Condition label 1: Act, label 2: Stim
behAll_HC.block = behAll_HC.block.replace(['Act', 'Stim'], [1, 2])
# number of participant
nParts_HC = len(np.unique(behAll_HC.sub_ID))
# participant indeces
behAll_HC.sub_ID = behAll_HC.sub_ID.replace(np.unique(behAll_HC.sub_ID), np.arange(1, nParts_HC +1))

"""select PD group"""
behAll_PD = behAll[behAll['patient']=='PD']
# Number of Medication effect (OFF vs ON) 
nMeds_PD = 2
# Number of conditions (Action vs Color)
nConds_PD = 2
# group label 1: PD OFF, group label 3: PD ON
behAll_PD['medication'] = behAll_PD.group.replace([1, 3], [1, 2])
# Condition label 1: Act, label 2: Stim
behAll_PD.block = behAll_PD.block.replace(['Act', 'Stim'], [1, 2])
# number of participant
nParts_PD = len(np.unique(behAll_PD.sub_ID))
# participant indeces
behAll_PD.sub_ID = behAll_PD.sub_ID.replace(np.unique(behAll_PD.sub_ID), np.arange(1, nParts_PD +1))

# Put required data for stan model
dataStan = {'N_HC':behAll_HC.shape[0],  
            'nParts_HC':nParts_HC, 
            'pushed_HC':np.array(behAll_HC.pushed).astype(int),  
            'yellowChosen_HC':np.array(behAll_HC.yellowChosen).astype(int), 
            'winAmtPushable_HC':np.array(behAll_HC.winAmtPushable), 
            'winAmtPullable_HC':np.array(behAll_HC.winAmtPullable),
            'winAmtYellow_HC':np.array(behAll_HC.winAmtYellow), 
            'winAmtBlue_HC':np.array(behAll_HC.winAmtPullable),
            'rewarded_HC':np.array(behAll_HC.correctChoice).astype(int), 
            'participant_HC':np.array(behAll_HC.sub_ID).astype(int),      
            'indicator_HC':np.array(behAll_HC.indicator).astype(int),  
            'nMeds_nSes_HC':nSes_HC,
            'nConds_HC':nConds_HC,
            'medication_session_HC':np.array(behAll_HC.session).astype(int),
            'condition_HC':np.array(behAll_HC.block).astype(int),
            'N_PD':behAll_PD.shape[0],  
            'nParts_PD':nParts_PD,  
            'pushed_PD':np.array(behAll_PD.pushed).astype(int),  
            'yellowChosen_PD':np.array(behAll_PD.yellowChosen).astype(int),  
            'winAmtPushable_PD':np.array(behAll_PD.winAmtPushable), 
            'winAmtPullable_PD':np.array(behAll_PD.winAmtPullable),
            'winAmtYellow_PD':np.array(behAll_PD.winAmtYellow), 
            'winAmtBlue_PD':np.array(behAll_PD.winAmtPullable),
            'rewarded_PD':np.array(behAll_PD.correctChoice).astype(int),   
            'participant_PD':np.array(behAll_PD.sub_ID).astype(int),      
            'indicator_PD':np.array(behAll_PD.indicator).astype(int),  
            'nMeds_nSes_PD':nMeds_PD,
            'nConds_PD':nConds_PD,
            'medication_session_PD':np.array(behAll_PD.medication).astype(int),
            'condition_PD':np.array(behAll_PD.block).astype(int),
            'p_push_init':.5, 
            'p_yell_init':.5,} 
 

# main directory of saving
mainScarch = '/mnt/scratch/projects/7TPD/amin'
# two HC and PD groups
partcipant_group = 'HC_PD'
# The adrees name of pickle file
pickelDir = f'{mainScarch}/realdata/hier/{partcipant_group}/{model_name}.pkl'
if modelFit == True: 
    """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
    # initial sampling
    initials = [] 
    for c in range(0, n_chains):
        chaininit = {
            'z_alphaAct_pos_HC': np.random.uniform(-1, 1, size=(nParts_HC, nSes_HC, nConds_HC)),
            'z_alphaAct_neg_HC': np.random.uniform(-1, 1, size=(nParts_HC, nSes_HC, nConds_HC)),
            'z_alphaClr_pos_HC': np.random.uniform(-1, 1, size=(nParts_HC, nSes_HC, nConds_HC)),
            'z_alphaClr_neg_HC': np.random.uniform(-1, 1, size=(nParts_HC, nSes_HC, nConds_HC)),
            'z_weight_HC': np.random.uniform(-1, 1, size=(nParts_HC, nSes_HC, nConds_HC)),
            'z_sensitivity_HC': np.random.uniform(-1, 1, size=(nParts_HC, nSes_HC, nConds_HC)),
            'hier_alpha_sd_HC': np.random.uniform(.01, .1),        
            'hier_weight_sd_HC': np.random.uniform(.01, .1),
            'hier_sensitivity_sd_HC': np.random.uniform(.01, .1),
            'z_alphaAct_pos_PD': np.random.uniform(-1, 1, size=(nParts_PD, nMeds_PD, nConds_PD)),
            'z_alphaAct_neg_PD': np.random.uniform(-1, 1, size=(nParts_PD, nMeds_PD, nConds_PD)),
            'z_alphaClr_pos_PD': np.random.uniform(-1, 1, size=(nParts_PD, nMeds_PD, nConds_PD)),
            'z_alphaClr_neg_PD': np.random.uniform(-1, 1, size=(nParts_PD, nMeds_PD, nConds_PD)),
            'z_weight_PD': np.random.uniform(-1, 1, size=(nParts_PD, nMeds_PD, nConds_PD)),
            'z_sensitivity_PD': np.random.uniform(-1, 1, size=(nParts_PD, nMeds_PD, nConds_PD)),
            'hier_alpha_sd_PD': np.random.uniform(.01, .1),        
            'hier_weight_sd_PD': np.random.uniform(.01, .1),
            'hier_sensitivity_sd_PD': np.random.uniform(.01, .1),
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
    if not os.path.isdir(mainScarch  + f'/realdata/hier/{partcipant_group}'):
            os.makedirs(mainScarch  + f'/realdata/hier/{partcipant_group}') 
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']

"""Healthu Control plot"""
# Extracting posterior distributions for each of four main unkhown parameters
alphaAct_pos_HC = fit["transfer_hier_alphaAct_pos_mu_HC"] 
alphaAct_neg_HC = fit["transfer_hier_alphaAct_neg_mu_HC"] 
alphaClr_pos_HC = fit["transfer_hier_alphaClr_pos_mu_HC"] 
alphaClr_neg_HC = fit["transfer_hier_alphaClr_neg_mu_HC"] 
weight_HC = fit["transfer_hier_weight_mu_HC"] 
beta_HC = fit["transfer_hier_sensitivity_mu_HC"]

# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(20, 8), tight_layout=True)
rows = 3
columns = 2

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(weight_HC[0,0], kde=True, stat='density', bins=100)
sns.histplot(weight_HC[0,1], kde=True, stat='density', bins=100)
sns.histplot(weight_HC[1,0], kde=True, stat='density', bins=100)
sns.histplot(weight_HC[1,1], kde=True, stat='density', bins=100)
plt.title('Weighting parameter',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel('$w_{(A)}$',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.xlim(0, 1)
# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(beta_HC[0,0], kde=True, stat='density', bins=100)
sns.histplot(beta_HC[0,1], kde=True, stat='density', bins=100)
sns.histplot(beta_HC[1,0], kde=True, stat='density', bins=100)
sns.histplot(beta_HC[1,1], kde=True, stat='density', bins=100)
plt.title('Sensitivity',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$\beta$',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 

# Action Learning Rate
fig.add_subplot(rows, columns, 3)
sns.histplot(alphaAct_pos_HC[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_pos_HC[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_pos_HC[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_pos_HC[1,1], kde=True, stat='density', bins=100)
plt.title('Positive Action Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(A)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Action Learning Rate
fig.add_subplot(rows, columns, 4)
sns.histplot(alphaClr_pos_HC[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_pos_HC[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_pos_HC[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_pos_HC[1,1], kde=True, stat='density', bins=100)
plt.title('Positive Color Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(C)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Action Learning Rate
fig.add_subplot(rows, columns, 5)
sns.histplot(alphaAct_neg_HC[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_neg_HC[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_neg_HC[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_neg_HC[1,1], kde=True, stat='density', bins=100)
plt.title('Negative Action Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(A)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Action Learning Rate
fig.add_subplot(rows, columns, 6)
sns.histplot(alphaClr_neg_HC[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_neg_HC[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_neg_HC[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_neg_HC[1,1], kde=True, stat='density', bins=100)
plt.title('Negative Color Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(C)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Save figure of parameter distribution 
fig.savefig(f'{mainScarch}/realdata/hier/{partcipant_group}/{model_name}_PD.png', dpi=500)
 

"""Parkinsons' disease plot"""
# Extracting posterior distributions for each of four main unkhown parameters
alphaAct_pos_PD = fit["transfer_hier_alphaAct_pos_mu_PD"] 
alphaAct_neg_PD = fit["transfer_hier_alphaAct_neg_mu_PD"] 
alphaClr_pos_PD = fit["transfer_hier_alphaClr_pos_mu_PD"] 
alphaClr_neg_PD = fit["transfer_hier_alphaClr_neg_mu_PD"] 
weight_PD = fit["transfer_hier_weight_mu_PD"] 
beta_PD = fit["transfer_hier_sensitivity_mu_PD"]

# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(20, 8), tight_layout=True)
rows = 3
columns = 2

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(weight_PD[0,0], kde=True, stat='density', bins=100)
sns.histplot(weight_PD[0,1], kde=True, stat='density', bins=100)
sns.histplot(weight_PD[1,0], kde=True, stat='density', bins=100)
sns.histplot(weight_PD[1,1], kde=True, stat='density', bins=100)
plt.title('Weighting parameter', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlabel('$w_{(A)}$', fontsize=18)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(beta_PD[0,0], kde=True, stat='density', bins=100)
sns.histplot(beta_PD[0,1], kde=True, stat='density', bins=100)
sns.histplot(beta_PD[1,0], kde=True, stat='density', bins=100)
sns.histplot(beta_PD[1,1], kde=True, stat='density', bins=100)
plt.title('Sensitivity', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlabel(r'$\beta$', fontsize=18)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr']) 

# Action Learning Rate
fig.add_subplot(rows, columns, 3)
sns.histplot(alphaAct_pos_PD[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_pos_PD[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_pos_PD[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_pos_PD[1,1], kde=True, stat='density', bins=100)
plt.title('Positive Action Learning Rate', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlabel(r'$ \alpha_{(A)} $', fontsize=18)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr'])  
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)


# Action Learning Rate
fig.add_subplot(rows, columns, 4)
sns.histplot(alphaClr_pos_PD[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_pos_PD[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_pos_PD[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_pos_PD[1,1], kde=True, stat='density', bins=100)
plt.title('Positive Color Learning Rate', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlabel(r'$ \alpha_{(C)} $', fontsize=18)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr'])  
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Action Learning Rate
fig.add_subplot(rows, columns, 5)
sns.histplot(alphaAct_neg_PD[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_neg_PD[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_neg_PD[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaAct_neg_PD[1,1], kde=True, stat='density', bins=100)
plt.title('Negative Action Learning Rate', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlabel(r'$ \alpha_{(A)} $', fontsize=18)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1) 


# Action Learning Rate
fig.add_subplot(rows, columns, 6)
sns.histplot(alphaClr_neg_PD[0,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_neg_PD[0,1], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_neg_PD[1,0], kde=True, stat='density', bins=100)
sns.histplot(alphaClr_neg_PD[1,1], kde=True, stat='density', bins=100)
plt.title('Negative Color Learning Rate', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xlabel(r'$ \alpha_{(C)} $', fontsize=18)
plt.legend(['OFF-Act', 'OFF-Clr', 'ON-Act', 'ON-Clr'])  
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(0, 1)

# Save figure of parameter distribution 
fig.savefig(f'{mainScarch}/realdata/hier/{partcipant_group}/{model_name}_PD.png', dpi=500)
 