#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

import numpy as np
import pandas as pd
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pylab as plt
import os

# Main directory of the subject
readMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/'
# read clinical evaluation
clinical_evaluation = pd.read_csv(f'{readMainDirec}/clinical_evaluation.csv')
 
 
# Estimate the density 
def get_mode_density(values):
    kde = gaussian_kde(values)
    x_grid = np.linspace(min(values), max(values), 1000)
    return x_grid[np.argmax(kde(x_grid))]


################################ model in PD
pickelDir_PD = f'/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Behavioral_Modeling/Tabel3/PD/tabel3_model1_complement_prob_PD.pkl'
loadPkl_PD = utils.load_pickle(load_path=pickelDir_PD)
fit_PD = loadPkl_PD['fit']
  
# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_alpha_pos_PD = fit_PD["transfer_alpha_pos"] 
transfer_alpha_neg_PD = fit_PD["transfer_alpha_neg"] 
transfer_sensitivity_PD = fit_PD["transfer_sensitivity"] 
transfer_weight_PD = fit_PD["transfer_weight"]
 


# dimetion of array
nParts= transfer_alpha_pos_PD.shape[0]
nMeds= transfer_alpha_pos_PD.shape[1]
nConds = 2
# Initialize array for modes
map_alpha_pos_PD = np.zeros((nParts, nMeds))
map_alpha_neg_PD = np.zeros((nParts, nMeds))
map_sensitivity_PD = np.zeros((nParts, nMeds))
for i in range(nParts):
    for j in range(nMeds):
        map_alpha_pos_PD[i,j] = get_mode_density(transfer_alpha_pos_PD[i,j]) 
        map_alpha_neg_PD[i,j] = get_mode_density(transfer_alpha_neg_PD[i,j]) 
        map_sensitivity_PD[i,j] = get_mode_density(transfer_sensitivity_PD[i,j]) 

# seighting parameters
map_weighting_PD = np.zeros((nParts, nConds, nMeds))
for i in range(nParts):
    for j in range(nConds):
        for k in range(nMeds):
            map_weighting_PD[i,j, k] = get_mode_density(transfer_weight_PD[i,j,k]) 


# PD medication effect in positive learning rate
map_med_alpha_pos_PD = map_alpha_pos_PD[:,1] - map_alpha_pos_PD[:,0]
# PD in positive learning rate
map_mean_alpha_pos_PD = np.mean([map_alpha_pos_PD[:,1], map_alpha_pos_PD[:,0]], axis=0)

# PD medication effect in negative learning rate
map_med_alpha_neg_PD = map_alpha_neg_PD[:,1]- map_alpha_neg_PD[:,0]
# PD in negative learning rate
map_mean_alpha_neg_PD = np.mean([map_alpha_neg_PD[:,1], map_alpha_neg_PD[:,0]], axis=0)

# PD medication effect in sensitivity
map_med_sensitivity_PD = map_sensitivity_PD[:,1]- map_sensitivity_PD[:,0]
# PD in sensitivity
map_mean_sensitivity_PD = np.mean([map_sensitivity_PD[:,1], map_sensitivity_PD[:,0]], axis=0)

# PD medication effect in weighting parameter in action value learning
map_med_weighting_act_PD = map_weighting_PD[:,0, 1] - map_weighting_PD[:,0, 0]
# PD in weighting parameter in action value learning
map_mean_weighting_act_PD = np.mean([map_weighting_PD[:,0, 1], map_weighting_PD[:,0, 0]], axis=0)

# PD medication effect in weighting parameter in color value learning
map_med_weighting_clr_PD = map_weighting_PD[:,1, 1] - map_weighting_PD[:,1, 0]
# PD in weighting parameter in color value learning
map_mean_weighting_clr_PD = np.mean([map_weighting_PD[:,1, 1], map_weighting_PD[:,1, 0]], axis=0)

# PD medication effect in weighting parameter across both action and color value learning
map_med_weighting_PD = (map_weighting_PD[:,0, 1] - map_weighting_PD[:,0, 0]) - (map_weighting_PD[:,1, 1] - map_weighting_PD[:,1, 0])
# PD in weighting parameter across both action and color value learning
map_mean_weighting_PD = np.mean([map_weighting_PD[:,0, 1],map_weighting_PD[:,0, 0], 1- map_weighting_PD[:,1, 1], 1-map_weighting_PD[:,1, 0]], axis=0)

################################ model in HC
pickelDir_HC = f'/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Behavioral_Modeling/Tabel3/HC/tabel3_model1_complement_prob_HC.pkl'
loadPkl_HC = utils.load_pickle(load_path=pickelDir_HC)
fit_HC = loadPkl_HC['fit']
  
# Extracting posterior distributions for each of four main unkhown parameters in HC
transfer_alpha_pos_HC = fit_HC["transfer_alpha_pos"] 
transfer_alpha_neg_HC = fit_HC["transfer_alpha_neg"] 
transfer_sensitivity_HC = fit_HC["transfer_sensitivity"] 
transfer_weight_HC = fit_HC["transfer_weight"]
 


# dimetion of array
nParts= transfer_alpha_pos_HC.shape[0]
nMeds= transfer_alpha_pos_HC.shape[1]
nConds = 2
# Initialize array for modes
map_alpha_pos_HC = np.zeros((nParts, nMeds))
map_alpha_neg_HC = np.zeros((nParts, nMeds))
for i in range(nParts):
    for j in range(nMeds):
        map_alpha_pos_HC[i,j] = get_mode_density(transfer_alpha_pos_HC[i,j]) 
        map_alpha_neg_HC[i,j] = get_mode_density(transfer_alpha_neg_HC[i,j]) 

# seighting parameters
map_weighting_HC = np.zeros((nParts, nConds, nMeds))
map_sensitivity_HC = np.zeros((nParts, nConds, nMeds))
for i in range(nParts):
    for j in range(nConds):
        for k in range(nMeds):
            map_weighting_HC[i,j, k] = get_mode_density(transfer_weight_HC[i,j,k]) 
            map_sensitivity_HC[i,j,k] = get_mode_density(transfer_sensitivity_HC[i,j, k]) 

# mean of learning rate across session 1 and 2
map_mean_alpha_pos_HC = np.mean([map_alpha_pos_HC[:,1],map_alpha_pos_HC[:,0]], axis=0)
map_mean_alpha_neg_HC = np.mean([map_alpha_neg_HC[:,1],map_alpha_neg_HC[:,0]], axis=0) 
# mean of sensitivty across session 1 and 2
map_mean_sensitivity_HC = np.mean([map_sensitivity_HC[:,0,0],map_sensitivity_HC[:,0,1],map_sensitivity_HC[:,1,0],map_sensitivity_HC[:,1,1],], axis=0)  
# mean of weighting across session 1 and 2 inn action value learning
map_mean_weighting_act_HC = np.mean([map_weighting_HC[:,0, 1], map_weighting_HC[:,0, 0]], axis=0)
# mean of weighting across session 1 and 2 inn color value learning
map_mean_weighting_clr_HC = np.mean([map_weighting_HC[:,1, 1], map_weighting_HC[:,1, 0]], axis=0)
# mean of weighting across session 1 and 2 in both action and color value learning
map_mean_weighting_HC = np.mean([map_weighting_HC[:,0, 1],map_weighting_HC[:,0, 0], 1- map_weighting_HC[:,1, 1], 1-map_weighting_HC[:,1, 0]], axis=0)


#### add new columns to clinical_evaluation
parameter_clinical_evaluation = clinical_evaluation.copy()

# Create masks
mask_HC = parameter_clinical_evaluation['group'] == 'HC'
mask_PD = parameter_clinical_evaluation['group'] == 'PD'
 
# mean positive alpha
parameter_clinical_evaluation.loc[mask_HC, 'map_mean_alpha_pos']=map_mean_alpha_pos_HC
parameter_clinical_evaluation.loc[mask_PD, 'map_mean_alpha_pos']=map_mean_alpha_pos_PD

# mean negative alpha
parameter_clinical_evaluation.loc[mask_HC, 'map_mean_alpha_neg']=map_mean_alpha_neg_HC
parameter_clinical_evaluation.loc[mask_PD, 'map_mean_alpha_neg']=map_mean_alpha_neg_PD
 
# mean sensitivity
parameter_clinical_evaluation.loc[mask_HC, 'map_mean_sensitivity']=map_mean_sensitivity_HC
parameter_clinical_evaluation.loc[mask_PD, 'map_mean_sensitivity']=map_mean_sensitivity_PD

# mean weighting in action value learning
parameter_clinical_evaluation.loc[mask_HC, 'map_mean_weighting_act']=map_mean_weighting_act_HC
parameter_clinical_evaluation.loc[mask_PD, 'map_mean_weighting_act']=map_mean_weighting_act_PD

# mean weighting in color value learning
parameter_clinical_evaluation.loc[mask_HC, 'map_mean_weighting_clr']=map_mean_weighting_clr_HC
parameter_clinical_evaluation.loc[mask_PD, 'map_mean_weighting_clr']=map_mean_weighting_clr_PD

# medication effect PD
parameter_clinical_evaluation.loc[mask_PD, 'map_med_alpha_pos']=map_med_alpha_pos_PD
parameter_clinical_evaluation.loc[mask_PD, 'map_med_alpha_neg']=map_med_alpha_neg_PD
parameter_clinical_evaluation.loc[mask_PD, 'map_med_sensitivity']=map_med_sensitivity_PD
parameter_clinical_evaluation.loc[mask_PD, 'map_med_weighting_act']=map_med_weighting_act_PD
parameter_clinical_evaluation.loc[mask_PD, 'map_med_weighting_clr']=map_med_weighting_clr_PD
parameter_clinical_evaluation.loc[mask_PD, 'map_med_weighting']=map_med_weighting_PD

 
# csv save colinical evaluation
if not os.path.isdir('/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/'):
        os.makedirs('/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/') 
parameter_clinical_evaluation.to_csv('/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/parameter_clinical_evaluation.csv', index=False)