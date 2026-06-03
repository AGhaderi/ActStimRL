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
model_calss ='tabel3'
# full model name
model_full_name = f'{model_calss}_{model}'

# if model will be fit or not
modelFit = False
# The adrees name of pickle file
maindir =f'{SCRATCH_HIER_MODEL_DIR}/{model_calss}/{model}/{partcipant_group}'
pickelDir = f'{maindir}/{model_full_name}_{partcipant_group}.pkl'
print(pickelDir)
# Check out if it does not exist
if not os.path.isdir(f'{maindir}/'):
        os.makedirs(f'{maindir}/') 

#Fitting data to model and then save as pickle file in the subject directory
if modelFit == True:     
    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/stan_models/hier_RL_cond_medSess.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # reading datastan for each model
    data_stan = dataStanActClr(readBehFile=PROJECT_NoNAN_BEH_ALL_FILE, group=partcipant_group, 
                               table=model_calss, model=model)
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = data_stan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=N_CHAIN, num_samples=N_SAMPLES, num_warmup=N_WARMUP)
    # Save Model Fit
    to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    #Loading the pickle file of model fit from the subject directory if modelFit = False
    loadPkl = load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']
 
# configuration, list of dictionary
config_hier, config_indv = config_plot_model(model_calss=model_calss, model_name=model, group=partcipant_group)

# plot the hierarchical posterior parameters
plot_hier_kde_posterior(fit=fit, model_dir=maindir, config=config_hier, group=partcipant_group, model_name=model)

# plot the individual posterior parameters seperate
plot_indv_kde_posterior_seperate(fit=fit, model_dir=maindir, config=config_indv, group=partcipant_group, model_name=model)

# plot the individual posterior parameters
save_indv_summary_posterior(fit=fit, param='weight', model_dir=maindir, group=partcipant_group, model_name=model)

