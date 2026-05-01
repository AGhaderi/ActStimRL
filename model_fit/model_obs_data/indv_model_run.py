#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

import numpy as np 
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from utils import *
import os

# group, PD, HC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
partcipant_group = 'PD' 
# full model name
model_full_name = 'indv_model1'

# if model will be fit or not
modelFit = False
# The adrees name of pickle file
maindir =f'{SCRATCH_INDV_MODEL_DIR}/{model_full_name}/{partcipant_group}/'
pickelDir = f'{maindir}/{model_full_name}_{partcipant_group}.pkl'
# Check out if it does not exist
if not os.path.isdir(f'{maindir}/'):
        os.makedirs(f'{maindir}/') 

#Fitting data to model and then save as pickle file in the subject directory
if modelFit == True:     
    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/stan_models/individual/{model_full_name}.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStanActClr(readBehFile=PROJECT_NoNAN_BEH_ALL_FILE, group=partcipant_group))
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=N_CHAIN, num_samples=N_SAMPLES, num_warmup=N_WARMUP)
    # Save Model Fit
    to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    #Loading the pickle file of model fit from the subject directory if modelFit = False
    loadPkl = load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']
 
# configuration, list of dictionary
if model_full_name=='indv_model1' and partcipant_group=='HC':
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Sess1-Act", "Weighting in Sess2-Act", "Weighting in Sess1-Clr", "Weighting in Sess2-Clr"], "range":(0,1)},
                   {"param": "weight", "label": ["Weighting in Sess1-Act", "Weighting in Sess2-Act", "Weighting in Sess1-Clr", "Weighting in Sess2-Clr"], "range":None},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Sess1-Act", "Positive learning rate in Sess2-Act", "Positive learning rate in Sess1-Clr", "Positive learning rate in Sess2-Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Sess1-Act", "Negative learning rate in Sess2-Act", "Negative learning rate in Sess1-Clr", "Negative learning rate in Sess2-Clr"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity in Sess1-Act", "Sensitivity in Sess2-Act", "Sensitivity in Sess1-Clr", "Sensitivity in Sess2-Clr"], "range":(0,.3)}]

if model_full_name=='indv_model1' and partcipant_group=='PD':
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in OFF-Act", "Weighting in ON-Act", "Weighting in OFF-Clr", "Weighting in ON-Clr"], "range":(0,1)},
                   {"param": "weight", "label": ["Weighting in OFF-Act", "Weighting in ON-Act", "Weighting in OFF-Clr", "Weighting in ON-Clr"], "range":None},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate in OFF-Act", "Positive learning rate in ON-Act", "Positive learning rate in OFF-Clr", "Positive learning rate in ON-Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate in OFF-Act", "Negative learning rate in ON-Act", "Negative learning rate in OFF-Clr", "Negative learning rate in ON-Clr"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity in OFF-Act", "Sensitivity in ON-Act", "Sensitivity in OFF-Clr", "Sensitivity in ON-Clr"], "range":(0,.3)}]
 
if model_full_name=='indv_model2' and partcipant_group=='HC':
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Sess1-Act", "Weighting in Sess2-Act", "Weighting in Sess1-Clr", "Weighting in Sess2-Clr"], "range":(0,1)},
                   {"param": "weight", "label": ["Weighting in Sess1-Act", "Weighting in Sess2-Act", "Weighting in Sess1-Clr", "Weighting in Sess2-Clr"], "range":None},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity"], "range":(0,.3)}]


if model_full_name=='indv_model2' and partcipant_group=='PD':
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in OFF-Act", "Weighting in ON-Act", "Weighting in OFF-Clr", "Weighting in ON-Clr"], "range":(0,1)},
                   {"param": "weight", "label": ["Weighting in OFF-Act", "Weighting in ON-Act", "Weighting in OFF-Clr", "Weighting in ON-Clr"], "range":None},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity"], "range":(0,.3)}]
    
# plot the individual posterior parameters, all particiapnts pool over
plot_indv_kde_posterior(fit=fit, dir=maindir, config=config_indv, group=partcipant_group, model=model_full_name)

# plot the individual posterior parameters, seperate for each particiapnts
plot_indv_kde_posterior_seperate(fit=fit, dir=maindir, config=config_indv, group=partcipant_group, model=model_full_name)

# save weihgintg paramters
save_indv_mean_posterior(fit=fit, main_dir=maindir, param='weight', group=partcipant_group, model=model_full_name)
