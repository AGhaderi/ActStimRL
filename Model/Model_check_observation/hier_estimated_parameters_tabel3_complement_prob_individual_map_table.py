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
from scipy.stats import gaussian_kde

def MAP(posterior_samples):
    # Estimate density
    kde = gaussian_kde(posterior_samples)

    # Evaluate KDE on a grid
    x = np.linspace(min(posterior_samples), max(posterior_samples), 1000)
    density = kde(x)

    # MAP estimate = location of the maximum density
    map_estimate = x[np.argmax(density)]
    return map_estimate, np.max(density)

# Read data fram of participatns
df = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/NoNanBehAll.csv')
 

# wirtten main directory  
writeMainScarch = '/mnt/scratch/projects/7TPD/amin'
# name of model
model_name = 'tabel3_model1_complement_prob'
# The adrees name of pickle file
pickelDir_HC = f'{writeMainScarch}/Behavioral/Tabel3/HC/tabel3_model1_complement_prob_HC.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{writeMainScarch}/Behavioral/Tabel3/PD/tabel3_model2_complement_prob_PD.pkl'
print(0)
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_HC = utils.load_pickle(load_path=pickelDir_HC)
loadPkl_PD = utils.load_pickle(load_path=pickelDir_PD)
fit_HC = loadPkl_HC['fit']
fit_PD = loadPkl_PD['fit']

print(1)
#####################Hierarchical Parameters
# Healthy Control
transfer_hier_alphaAct_pos_mu_HC = fit_HC["transfer_hier_alphaAct_pos_mu"] 
transfer_hier_alphaAct_neg_mu_HC = fit_HC["transfer_hier_alphaAct_neg_mu"] 
transfer_hier_alphaClr_pos_mu_HC = fit_HC["transfer_hier_alphaClr_pos_mu"] 
transfer_hier_alphaClr_neg_mu_HC = fit_HC["transfer_hier_alphaClr_neg_mu"] 
transfer_hier_weight_mu_HC = fit_HC["transfer_hier_weight_mu"] 
transfer_hier_sensitivity_mu_HC = fit_HC["transfer_hier_sensitivity_mu"]

# Parkinson's disease
transfer_hier_alphaAct_pos_mu_PD = fit_PD["transfer_hier_alphaAct_pos_mu"] 
transfer_hier_alphaAct_neg_mu_PD = fit_PD["transfer_hier_alphaAct_neg_mu"] 
transfer_hier_alphaClr_pos_mu_PD = fit_PD["transfer_hier_alphaClr_pos_mu"] 
transfer_hier_alphaClr_neg_mu_PD = fit_PD["transfer_hier_alphaClr_neg_mu"] 
transfer_hier_weight_mu_PD = fit_PD["transfer_hier_weight_mu"] 
transfer_hier_sensitivity_mu_PD = fit_PD["transfer_hier_sensitivity_mu"]



####################################Individual Parameters

# Healthy Control
transfer_weight_HC=fit_HC['transfer_weight']
transfer_alphaAct_pos_HC=fit_HC['transfer_alphaAct_pos']
transfer_alphaAct_neg_HC=fit_HC['transfer_alphaAct_neg']
transfer_alphaClr_pos_HC=fit_HC['transfer_alphaClr_pos']
transfer_alphaClr_neg_HC=fit_HC['transfer_alphaClr_neg']
transfer_sensitivity_HC=fit_HC['transfer_sensitivity']


# Healthy Control
transfer_weight_PD=fit_PD['transfer_weight']
transfer_alphaAct_pos_PD=fit_PD['transfer_alphaAct_pos']
transfer_alphaAct_neg_PD=fit_PD['transfer_alphaAct_neg']
transfer_alphaClr_pos_PD=fit_PD['transfer_alphaClr_pos']
transfer_alphaClr_neg_PD=fit_PD['transfer_alphaClr_neg']
transfer_sensitivity_PD=fit_PD['transfer_sensitivity']

# relevant columns for HC
df_HC = df[(df['patient']=='HC')][['session', 'block', 'sub_ID', 'patient']].drop_duplicates().reset_index()
df_HC[['weight_map', 'sensivity_map']] = None
# relevant columns for PD
df_PD = df[(df['patient']=='PD')][['session', 'block', 'sub_ID', 'patient']].drop_duplicates().reset_index()
df_PD[['weight_map', 'sensivity_map']] = None
# number of participants
parts_HC = df_HC['sub_ID'].unique()
parts_PD = df_PD['sub_ID'].unique()


# adding map for HC
for i, part in enumerate(parts_HC):
    for s in [1, 2]:
        for c, condName in enumerate(['Act', 'Stim']):
            map_apex, dens_apex = MAP(transfer_weight_HC[i,s-1,c])
            df_HC.loc[(df_HC['sub_ID']==part)& (df_HC['session']==s)&(df_HC['block']==condName), 'weight_map'] = map_apex
            # calculate MAP for sensitivty parameter
            map_apex, dens_apex = MAP(transfer_sensitivity_HC[i,s-1,c])
            df_HC.loc[(df_HC['sub_ID']==part)& (df_HC['session']==s)&(df_HC['block']==condName), 'sensivity_map'] = map_apex
# adding map for PD
for i, part in enumerate(parts_PD):
    for s in [1, 2]:
        for c, condName in enumerate(['Act', 'Stim']):
            # calculate MAP for weighting parameter
            map_apex, dens_apex = MAP(transfer_weight_PD[i,s-1,c])
            df_PD.loc[(df_PD['sub_ID']==part)& (df_PD['session']==s)&(df_PD['block']==condName), 'weight_map'] = map_apex
        # calculate MAP for sensitivity parameter
        map_apex, dens_apex = MAP(transfer_sensitivity_PD[i,s-1])
        df_PD.loc[(df_PD['sub_ID']==part)& (df_PD['session']==s), 'sensivity_map'] = map_apex

# mergen two data frame HC and PD
df_out = pd.concat([df_HC, df_PD], ignore_index=True) 

df_out.to_csv('/mnt/scratch/projects/7TPD/amin/beh_map.csv')