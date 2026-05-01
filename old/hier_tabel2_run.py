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

# name of the model
model = 'model1'
# group, PD, HC
partcipant_group = 'PD' 
# class of model
model_calss ='tabel2'
# full model name
model_full_name = f'{model_calss}_{model}'

# if model will be fit or not
modelFit = False
# The adrees name of pickle file
pickelDir = f'{SCRATCH_HIER_MODEL_DIR}/{model_calss}/{partcipant_group}/{model_full_name}_{partcipant_group}.pkl'
# Check out if it does not exist
if not os.path.isdir(f'{SCRATCH_HIER_MODEL_DIR}/{model_calss}/{partcipant_group}/'):
        os.makedirs(f'{SCRATCH_HIER_MODEL_DIR}/{model_calss}/{partcipant_group}/') 

#Fitting data to model and then save as pickle file in the subject directory
if modelFit == True:     
    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/stan_models/{model_calss}/{model_full_name}.stan' 
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
if model=='model1':
    #hierarchical
    config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]
    #individual
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Act", "Positive learning rate in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act", "Negative learning rate in Clr"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]


if model=='model2':
    config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": None, "range":(0,1)},
                   {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": None, "range":(0,1)},
                   {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]
    #individual
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]


if model=='model3':
    config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": None, "range":(0,.1)}]
    #individual
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Act", "Positive learning rate in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act", "Negative learning rate in Clr"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity"], "range":(0,.3)}]

if model=='model4':
    config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": None, "range":(0,1)},
                   {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]

    #individual
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Act", "Positive learning rate in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]

if model=='model5':
    config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": None, "range":(0,1)},
                   {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]

    #individual
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act", "Negative learning rate in Clr"], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]

if model=='model6':
    config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                   {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": None, "range":(0,1)},
                   {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": None, "range":(0,1)},
                   {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": None, "range":(0,.1)}]
        
    #individual
    config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                   {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                   {"param": "transfer_alpha_neg", "label": ["Negative learning rate",], "range":(0,1)},
                   {"param": "transfer_sensitivity", "label": ["Sensitivity"], "range":(0,.3)}]



# plot the hierarchical posterior parameters
plot_hier_kde_posterior(fit=fit, config=config_hier, group=partcipant_group, model_name=model_full_name, model_calss=model_calss)


# plot the individual posterior parameters
plot_indv_kde_posterior(fit=fit, config=config_indv, group=partcipant_group, model_name=model_full_name, model_calss=model_calss)