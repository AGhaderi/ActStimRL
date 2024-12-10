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

# session effect over Parkinsdon's disease
partcipant_group = 'PD'
# Get the filename of the currently running script
filename = os.path.basename(__file__)
# Remove the .py extension from the filename
model_name = os.path.splitext(filename)[0]
 
# If you want to model fit or just recall ex model fit
modelFit = True
# Number of chains in MCMC procedure
n_chains = 4
# The number of iteration or samples for each chain in MCM procedure
n_samples=4000
# Directory of simulated data
parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation'
# number of simulation
simNumber = 3
# read collected data across all participants
behAll = pd.read_csv(f'{parent_dir}/{str(simNumber)}/{str(simNumber)}-all-sim-task-design-true-param.csv')
# list of subjects
subList = behAll['sub_ID'].unique()

# set of indicator to the first trial of each participant
for sub in subList:
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Stim']: # condition
                behAll_indicator = behAll[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse)]  
                behAll.loc[(behAll['sub_ID']==sub)&(behAll['block']==condition)&(behAll['session']==session)&(behAll['reverse']==reverse), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)

# select PD group
behAll = behAll[behAll['patient']==partcipant_group]

#  set the session or medication effect
medication_session = np.array(behAll.session).astype(int)

# Number of session 1 and 2
nMeds_nSes = 2
# Number of conditions (Action vs Color)
nConds = 2
# Condition label 1: Act, label 2: Stim
behAll.block = behAll.block.replace(['Act', 'Stim'], [1, 2])
# number of participant
nParts = len(np.unique(behAll.sub_ID))
# participant indeces
behAll.sub_ID = behAll.sub_ID.replace(np.unique(behAll.sub_ID), np.arange(1, nParts +1))
# The adrees name of pickle file
pickelDir = f'{parent_dir}/{str(simNumber)}/{partcipant_group}/{model_name}.pkl'
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
                'nMeds_nSes':nMeds_nSes,
                'nConds':nConds,
                'medication_session':medication_session,
                'condition':np.array(behAll.block).astype(int),
                'p_push_init':.5, 
                'p_yell_init':.5}
    # initial sampling
    initials = [] 
    for c in range(0, n_chains):
        chaininit = {
            'z_alphaAct_pos': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_alphaAct_neg': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_alphaClr_pos': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_alphaClr_neg': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_weight': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'z_sensitivity': np.random.uniform(-1, 1, size=(nParts, nMeds_nSes, nConds)),
            'hier_alpha_sd': np.random.uniform(.01, .1),        
            'hier_weight_sd': np.random.uniform(.01, .1),
            'hier_sensitivity_sd': np.random.uniform(.01, .1),
        }
        initials.append(chaininit)   

    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/Model/stan_models/{model_name}.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
    nest_asyncio.apply()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init=initials)
    # Save Model Fit
    if not os.path.isdir(f'{parent_dir}/{str(simNumber)}/{partcipant_group}/'):
            os.makedirs(f'{parent_dir}/{str(simNumber)}/{partcipant_group}/') 
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']

# Extracting posterior distributions for each of four main unkhown parameters
estimated_hier_alphaAct_pos_mu = fit["hier_alphaAct_pos_mu"] 
estimated_hier_alphaAct_neg_mu = fit["hier_alphaAct_neg_mu"] 
estimated_hier_alphaClr_pos_mu = fit["hier_alphaClr_pos_mu"] 
estimated_hier_alphaClr_neg_mu = fit["hier_alphaClr_neg_mu"] 
estimated_hier_weight_mu = fit["hier_weight_mu"] 
estimated_hier_sensitivity_mu = fit["hier_sensitivity_mu"]
estimated_hier_alpha_sd = fit['hier_alpha_sd'].flatten()
estimated_hier_weight_sd = fit['hier_weight_sd'].flatten()
estimated_hier_sensitivity_sd = fit['hier_sensitivity_sd'].flatten()

# grand truth
grand_trusth = pd.read_csv(f'{parent_dir}/{str(simNumber)}/hier-true-param.csv')
true_hier_alphaAct_pos_mu = np.array(grand_trusth["hier_alphaAct_pos_mu"]) 
true_hier_alphaAct_neg_mu = np.array(grand_trusth["hier_alphaAct_neg_mu"]) 
true_hier_alphaClr_pos_mu = np.array(grand_trusth["hier_alphaClr_pos_mu"]) 
true_hier_alphaClr_neg_mu = np.array(grand_trusth["hier_alphaClr_neg_mu"]) 
true_hier_weight_mu =  np.array(grand_trusth["hier_weight_mu"]) 
true_hier_sensitivity_mu =  np.array(grand_trusth["hier_sensitivity_mu"])
true_hier_alpha_sd =  np.array(grand_trusth['hier_alpha_sd'].unique())[0]
true_hier_weight_sd =  np.array(grand_trusth['hier_weight_sd'].unique())[0]
true_hier_sensitivity_sd =  np.array(grand_trusth['hier_sensitivity_sd'].unique())[0]

# Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(20, 12), tight_layout=True)
rows = 4
columns = 2

# Weghtening
fig.add_subplot(rows, columns, 1)
sns.histplot(estimated_hier_weight_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_weight_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_weight_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_weight_mu[1,1], kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- Weighting parameter',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel('$w_{(A)}$',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# Sensitivity
fig.add_subplot(rows, columns, 2)
sns.histplot(estimated_hier_sensitivity_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_sensitivity_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_sensitivity_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_sensitivity_mu[1,1], kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- Sensitivity',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$\beta$',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
# Action Learning Rate
fig.add_subplot(rows, columns, 3)
sns.histplot(estimated_hier_alphaAct_pos_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaAct_pos_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaAct_pos_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaAct_pos_mu[1,1], kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- Positive Action Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(A)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr'])
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# Action Learning Rate
fig.add_subplot(rows, columns, 4)
sns.histplot(estimated_hier_alphaClr_pos_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaClr_pos_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaClr_pos_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaClr_pos_mu[1,1], kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- Positive Color Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(C)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr'])
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# Action Learning Rate
fig.add_subplot(rows, columns, 5)
sns.histplot(estimated_hier_alphaAct_neg_mu[0,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaAct_neg_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaAct_neg_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaAct_neg_mu[1,1], kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- Negative Action Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(A)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# Action Learning Rate
fig.add_subplot(rows, columns, 6)
sns.histplot(estimated_hier_alphaClr_neg_mu[0,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaClr_neg_mu[1,0], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaClr_neg_mu[1,1], kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alphaClr_neg_mu[0,0], kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- Negative Color Learning Rate',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$ \alpha_{(C)} $',  fontsize=18)
plt.legend(['Sess1-Act', 'Sess1-Clr', 'Sess2-Act', 'Sess2-Clr']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)



# Action Learning Rate
fig.add_subplot(rows, columns, 7)
sns.histplot(estimated_hier_weight_sd, kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_alpha_sd, kde=True, stat='density', bins=100)
sns.histplot(estimated_hier_sensitivity_sd, kde=True, stat='density', bins=100)
plt.title(f'{partcipant_group}- standard deviation',  fontsize=18)
plt.ylabel('Density',  fontsize=18)
plt.xlabel(r'$STD$',  fontsize=18)
plt.legend(['Wighting', 'Learning rate', 'Sensitivity']) 
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# Save figure of parameter distribution 
fig.savefig(f'{parent_dir}/{str(simNumber)}/{partcipant_group}/{model_name}.png', dpi=500)
 
 