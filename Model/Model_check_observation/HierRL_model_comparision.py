#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
import arviz as az
from scipy import stats

######################################### Model Comparision for evaluating the effect of session
for partcipant_group in ['HC', 'PD']:
    # number of models
    list_model = ['HierRL_dual_diffLR_1', 'HierRL_dual_diffLR_2', 'HierRL_dual_diffLR_3', 'HierRL_dual_diffLR_4', 
                 'HierRL_dual_diffLR_5', 'HierRL_dual_diffLR_6', 'HierRL_dual_diffLR_7']
    # declare waice variable
    waic = np.zeros(len(list_model))
    lppd = np.zeros(len(list_model))
    # loop over list of participants
    for i, model in enumerate(list_model):
        print(model)
        # main directory of saving
        mainScarch = '/mnt/scratch/projects/7TPD/amin/'
        # pickle fine in the scratch folder
        pickelDir = f'{mainScarch}/realdata/{partcipant_group}/{model}.pkl'
        """Loading the pickle file of model fit from the subject directory"""
        loadPkl = utils.load_pickle(load_path=pickelDir)
        fit = loadPkl['fit'] 
        # get the linkelihood and comarision assessment       
        log_lik = fit['log_lik']
        print(log_lik.shape)
        model_Comparision_criteria = utils.waic(log_likelihood=log_lik)
        waic[i] = model_Comparision_criteria['waic']
        lppd[i] = model_Comparision_criteria['lppd']

    ## waic
    print(f'WAIC in {partcipant_group} for 7 model: ',np.round(waic))
    #dwaic
    dWAIC = np.round(waic - np.min(waic))
    print(f'dWAIC in {partcipant_group}  for 7 model: ', dWAIC)
    # realtive weight
    weight = [np.exp(-.5*dWAIC[i])/np.sum(np.exp(-.5*dWAIC)) for i in range(len(dWAIC))]
    print(f'weight in {partcipant_group}  for 7 model: ', np.round(weight))


######################################## Model Comparision for evaluting the different structure of hierachcial model

#for partcipant_group in ['HC']:
#    # number of models
#    list_model = ['HierRL_one_diffLR', 'HierRL_one_sameLR', 'HierRL_dual_diffLR_1']
#    # declare waice variable
#    waic = np.zeros(len(list_model))
#    lppd = np.zeros(len(list_model))
#    # loop over list of participants
#    for i, model in enumerate(list_model):
#        print(model)
#        # main directory of saving
#        mainScarch = '/mnt/scratch/projects/7TPD/amin/'
#        # pickle fine in the scratch folder
#        pickelDir = f'{mainScarch}/realdata/{partcipant_group}/{model}.pkl'
#        """Loading the pickle file of model fit from the subject directory"""
#        loadPkl = utils.load_pickle(load_path=pickelDir)
#        fit = loadPkl['fit'] 
#        # get the linkelihood and comarision assessment       
#        log_lik = fit['log_lik']
#        print(log_lik.shape)
#        model_Comparision_criteria = utils.waic(log_likelihood=log_lik)
#        waic[i] = model_Comparision_criteria['waic']
#        lppd[i] = model_Comparision_criteria['lppd']
#
#    ## waic
#    print(f'WAIC in {partcipant_group} for 7 model: ',np.round(waic))
#    #dwaic
#    dWAIC = np.round(waic - np.min(waic))
#    print(f'dWAIC in {partcipant_group}  for 7 model: ', dWAIC)
#    # realtive weight
#    weight = [np.exp(-.5*dWAIC[i])/np.sum(np.exp(-.5*dWAIC)) for i in range(len(dWAIC))]
#    print(f'weight in {partcipant_group}  for 7 model: ', np.round(weight))
