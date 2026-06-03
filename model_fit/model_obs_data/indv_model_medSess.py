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
model_name = 'model1'

# if model will be fit or not
modelFit = False
# The adrees name of pickle file
maindir_scratch =f'{SCRATCH_INDV_MODEL_DIR}/{model_name}/{partcipant_group}/'
maindir_project =f'{PROJECT_INDV_MODEL_DIR}/{model_name}/{partcipant_group}/'
pickelDir = f'{maindir_project}/{model_name}_{partcipant_group}.pkl'
# Check out if it does not exist
if not os.path.isdir(f'{maindir_scratch}/'):
        os.makedirs(f'{maindir_scratch}/') 

#Fitting data to model and then save as pickle file in the subject directory
if modelFit == True:     
    # Loading the RL Stan Model
    file_name = f'/mrhome/amingk/Documents/7TPD/ActStimRL/stan_models/indv_RL_cond_medSess.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStanActClr(readBehFile=PROJECT_NoNAN_BEH_ALL_FILE, group=partcipant_group,
                                                             table='tabel3', model=model_name))
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=N_CHAIN, num_samples=N_SAMPLES, num_warmup=N_WARMUP)
    # Save Model Fit
    to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    #Loading the pickle file of model fit from the subject directory if modelFit = False
    loadPkl = load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']
 
# configuration, list of dictionary
config_hier, config_indv = config_plot_model(model_calss='tabel3', model_name=model_name, 
                                             group=partcipant_group)

# plot the individual posterior parameters, all particiapnts pool over
plot_indv_kde_posterior(fit=fit, model_dir=maindir_scratch, config=config_indv, group=partcipant_group, model_name=model_name)

# plot the individual posterior parameters, seperate for each particiapnts
plot_indv_kde_posterior_seperate(fit=fit, model_dir=maindir_scratch, config=config_indv, group=partcipant_group, model_name=model_name)

# save weihgintg paramters
save_indv_summary_posterior(fit=fit, model_dir=maindir_scratch, param='weight', group=partcipant_group, model_name=model_name)


