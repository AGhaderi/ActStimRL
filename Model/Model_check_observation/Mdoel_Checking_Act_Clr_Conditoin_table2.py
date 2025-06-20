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

######################################## Model Comparision for evaluting the different structure of hierachcial model

for partcipant_group in ['HC', 'PD']:
    # name of table
    tabel = 'Table2'
    list_model = ['tabel2_model1_complement_prob', 'tabel2_model2_complement_prob', 'tabel2_model3_complement_prob',
                'tabel2_model4_complement_prob', 'tabel2_model5_complement_prob']
    # declare waice variable
    waic = np.zeros(len(list_model))
    log_lppd_models = np.zeros(len(list_model))
    # loop over list of participants
    for i, model_name in enumerate(list_model):
        print(model_name)
        # main directory of saving
        writeMainScarch = '/mnt/scratch/projects/7TPD/amin'
        # The adrees name of pickle file
        pickelDir = f'{writeMainScarch}/Behavioral/Tabel2/{partcipant_group}/{model_name}_{partcipant_group}.pkl'
        """Loading the pickle file of model fit from the subject directory"""
        loadPkl = utils.load_pickle(load_path=pickelDir)
        fit = loadPkl['fit'] 
        # get the linkelihood and comarision assessment       
        log_lik = fit['log_lik']
        print(log_lik.shape)
        model_Comparision_criteria = utils.waic(log_likelihood=log_lik)
        waic[i] = model_Comparision_criteria['waic']


    ## waic
    print(f'WAIC in {partcipant_group} for: ',tabel, ' : ', np.round(waic))
    #dwaic
    dWAIC = np.round(waic - np.min(waic))
    print(f'dWAIC in {partcipant_group}  for: ',tabel, ' : ',dWAIC)
    # realtive weight
    weight = [np.exp(-.5*dWAIC[i])/np.sum(np.exp(-.5*dWAIC)) for i in range(len(dWAIC))]
    print(f'weight in {partcipant_group}  for: ',tabel, ' : ', np.round(weight))
