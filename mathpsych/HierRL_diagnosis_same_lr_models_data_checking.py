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
import os
import plots

# select Act or Stim to model fit seperately
cond_act_stim = 'Act'
# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')
# select Action value learning and parkinsons disease
behAll = behAll[(behAll['block']==cond_act_stim)&(behAll['patient']=='PD')]
# the list of participant
subList_PD = np.unique(behAll['sub_ID'])
# number of models
n_models =  6
# declare waice variable
log_waic_models = np.zeros(n_models)
log_lppd_models = np.zeros(n_models)
# loop over list of participants
for m in range(2,n_models):
    print(f'Model {m}')
    # main directory of saving
    mainScarch = '/mnt/scratch/projects/7TPD/amin'
    # pickle fine in the scratch folder
    pickelDir = f'{mainScarch}/realdata/hier/PD-HC/diagnosis/HierRL_diagnosis_same_lr_model{m+1}_data_{cond_act_stim}.pkl'
    """Loading the pickle file of model fit from the subject directory"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit'] 
    # get the linkelihood and comarision assessment       
    log_lik = fit['log_lik']
    print(log_lik.shape)
    log_assessement = utils.waic(log_likelihood=log_lik)
    log_waic_models[m] = log_assessement['waic']
    log_lppd_models[m] = log_assessement['lppd']