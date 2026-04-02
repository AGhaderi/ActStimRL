#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

import numpy as np 
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
import os
from utils import model_utils
from utils import config

# Get the filename of the currently running script
filename = os.path.basename(__file__)
# Remove the .py extension from the filename
model_name = os.path.splitext(filename)[0]

# session effect over Parkinsdon's disease
partcipant_group = 'PD' 

# read collected data across all participants
behAll = pd.read_csv(config.PROJECT_NoNAN_BEH_ALL_FILE)

# select group 
behAll = behAll[(behAll['patient']==partcipant_group)].copy().reset_index(drop=False)

# number of participant
nParts = len(np.unique(behAll['sub_ID']))
# Convert participant ID to indeces
behAll['sub_ID'] = behAll['sub_ID'].replace(np.unique(behAll.sub_ID), np.arange(1, nParts +1))
# number of condition
nConds = 2
# Condition label 1: Act, label 2: Stim
behAll.block = behAll.block.replace(['Act', 'Stim'], [1, 2])
# Number of session 1 and 2
nMeds_nSes = 2
#  set the session or medication effect
if partcipant_group=='HC':
    medication_session = np.array(behAll.session).astype(int)
elif partcipant_group=='PD':
    # group label 1: PD OFF, group label 3: PD ON
    behAll['medication'] = behAll.group.replace([1, 3], [1, 2])
    medication_session = np.array(behAll.medication).astype(int)

# If you want to model fit or just recall ex model fit
modelFit = False
  
# The adrees name of pickle file
pickelDir = f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{partcipant_group}/{model_name}_{partcipant_group}.pkl'
# Check out if it does not exist
if not os.path.isdir(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{partcipant_group}/'):
        os.makedirs(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{partcipant_group}/') 

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
                'winAmtBlue':np.array(behAll.winAmtBlue),
                'rewarded':np.array(behAll.correctChoice).astype(int), # should be integer   
                'participant':np.array(behAll.sub_ID).astype(int),      
                'indicator':np.array(behAll.indicator).astype(int),
                'nConds':nConds,
                'condition':np.array(behAll.block).astype(int),
                'nMeds_nSes':nMeds_nSes,
                'medication_session':medication_session
                }
    # initial sampling
    initials = [] 
    for c in range(0, config.N_CHAIN):
        chaininit = {
            'transfer_alpha_pos': np.random.uniform(.4, .6, size=(nParts, nMeds_nSes)),
            'transfer_alpha_neg': np.random.uniform(.4, .6, size=(nParts, nConds, nMeds_nSes)),
            'transfer_sensitivity': np.random.uniform(.03, .07, size=(nParts, nConds, nMeds_nSes)),
            'transfer_weight':np.random.uniform(.4, .6, size=(nParts, nConds)),
            'hier_alpha_sd': np.random.uniform(.01, .1),        
            'hier_sensitivity_sd': np.random.uniform(.01, .02)
        }
        initials.append(chaininit)   
        
    # Loading the RL Stan Model
    file_name = f'{config.STAN_DIR}/MedSess_tabel3/{model_name}.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
 
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=config.N_CHAIN, num_samples=config.N_SAMPLES, init=initials, num_warmup=config.N_WARMUP)
    # Save Model Fit
    model_utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = model_utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']

 
# Extracting posterior distributions for each of four main unkhown parameters
hier_weight_mu = fit["transfer_hier_weight_mu"] 
hier_alpha_pos_mu = fit["transfer_hier_alpha_pos_mu"] 
hier_alpha_neg_mu = fit["transfer_hier_alpha_neg_mu"] 
hier_sensitivity_mu = fit["transfer_hier_sensitivity_mu"]  

# Convert hierarchical parameter plots to axs structure
mm = 1/2.54  # convert cm to inches
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(31*mm, 16*mm))
axs = axs.flatten()  # flatten to 1D array for easy indexing
 
# ------------------- Hierarchical Weighting -------------------
for i, dist in enumerate([hier_weight_mu[0,0], hier_weight_mu[0,1], hier_weight_mu[1,0], hier_weight_mu[1,1]]):
    sns.histplot(dist, kde=True, stat='density', bins=100, ax=axs[0])
axs[0].set_title('Hierarchical Weighting', fontsize=14)
axs[0].set_xlabel(r'$ w $', fontsize=12)
axs[0].set_ylabel('Density', fontsize=12)
axs[0].set_xlim(0, 1)
if partcipant_group=='HC':
    axs[0].legend(['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'])
elif partcipant_group=='PD':
    axs[0].legend(['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'])

# ------------------- Positive Learning Rate -------------------
for i, dist in enumerate([hier_alpha_pos_mu[0], hier_alpha_pos_mu[1]]):
    sns.histplot(dist, kde=True, stat='density', bins=100, ax=axs[1])
axs[1].set_title('Hierarchical Positive Learning Rate', fontsize=14)
axs[1].set_xlabel(r'$ +\alpha $', fontsize=12)
axs[1].set_ylabel('Density', fontsize=12)
axs[1].set_xlim(0, 1)
if partcipant_group=='HC':
    axs[1].legend(['Sess1', 'Sess2'])
elif partcipant_group=='PD':
    axs[1].legend(['OFF', 'ON'])

# ------------------- Negative Learning Rate -------------------
for i, dist in enumerate([hier_alpha_neg_mu[0,0], hier_alpha_neg_mu[0,1], hier_alpha_neg_mu[1,0], hier_alpha_neg_mu[1,1]]):
    sns.histplot(dist, kde=True, stat='density', bins=100, ax=axs[2])
axs[2].set_title('Hierarchical Negative Learning Rate', fontsize=14)
axs[2].set_xlabel(r'$ -\alpha $', fontsize=12)
axs[2].set_ylabel('Density', fontsize=12)
axs[2].set_xlim(0, 1)
if partcipant_group=='HC':
    axs[2].legend(['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'])
elif partcipant_group=='PD':
    axs[2].legend(['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'])

# ------------------- Sensitivity -------------------
for i, dist in enumerate([hier_sensitivity_mu[0,0], hier_sensitivity_mu[0,1], hier_sensitivity_mu[1,0], hier_sensitivity_mu[1,1]]):
    sns.histplot(dist, kde=True, stat='density', bins=100, ax=axs[3])
axs[3].set_title('Hierarchical Sensitivity', fontsize=14)
axs[3].set_xlabel(r'$\beta$', fontsize=12)
axs[3].set_ylabel('Density', fontsize=12)
if partcipant_group=='HC':
    axs[3].legend(['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'])
elif partcipant_group=='PD':
    axs[3].legend(['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'])

# Adjust layout and save
fig.tight_layout()
fig.savefig(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{partcipant_group}/{model_name}_{partcipant_group}.png', dpi=500)
plt.show()
