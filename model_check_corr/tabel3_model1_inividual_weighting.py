#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
import arviz as az
from scipy.stats import gaussian_kde
from utils import config, model_utils
import os

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
behAll = pd.read_csv(config.PROJECT_NoNAN_BEH_ALL_FILE)

# select group 
behAll_PD = behAll[(behAll['patient']=='PD')].copy().reset_index(drop=False)
participants_PD = np.unique(behAll_PD['sub_ID'])
nParts = len(np.unique(behAll_PD['sub_ID']))

# name of model
model_name = 'tabel3_model1'
# The adrees name of pickle file
pickelDir_HC = f'{config.PROJECT_HIER_MODEL_DIR}/Tabel3/HC/{model_name}_HC.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{config.PROJECT_HIER_MODEL_DIR}/Tabel3/PD/{model_name}_PD.pkl'
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_HC = model_utils.load_pickle(load_path=pickelDir_HC)
loadPkl_PD = model_utils.load_pickle(load_path=pickelDir_PD)
fit_HC = loadPkl_HC['fit']
fit_PD = loadPkl_PD['fit']

# Extracting posterior distributions in HC
hier_weight_mu_HC = fit_HC["transfer_hier_weight_mu"] 
hier_alpha_pos_mu_HC = fit_HC["transfer_hier_alpha_pos_mu"]
hier_alpha_neg_mu_HC = fit_HC["transfer_hier_alpha_neg_mu"]
hier_sensitivity_mu_HC = fit_HC["transfer_hier_sensitivity_mu"] 

weight_HC = fit_HC["transfer_weight"] 
alpha_pos_HC = fit_HC["transfer_alpha_pos"]
alpha_neg_HC = fit_HC["transfer_alpha_neg"]
sensitivity_HC = fit_HC["transfer_sensitivity"] 

# Healthy Control

# Figure
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(30*mm, 30*mm))
axs = axs.flatten()

# action calaue learning
for i in range(weight_HC.shape[0]):
    # Act-Session 1
    sns.kdeplot(data=weight_HC[i,0,0], ax=axs[0], alpha=1)
    axs[0].set_xlim(0,1)
    axs[0].set_title('Act-Sess1-HC, weighting parameter')

    # Act-Session 2
    sns.kdeplot(data=weight_HC[i,0,1], ax=axs[1], alpha=1)
    axs[1].set_xlim(0,1)
    axs[1].set_title('Act-Sess2-HC, weighting parameter')


    # Clr-Session 1
    sns.kdeplot(data=weight_HC[i,1,0], ax=axs[2], alpha=1)
    axs[2].set_xlim(0,1)
    axs[2].set_title('Clr-Sess1-HC, weighting parameter')

    # Act-Session 2
    sns.kdeplot(data=weight_HC[i,1,1], ax=axs[3], alpha=1)
    axs[3].set_xlim(0,1)
    axs[3].set_title('Clr-Sess1-HC, weighting parameter')

# hierarchical parameter
sns.kdeplot(data=hier_weight_mu_HC[0,0], ax=axs[0], multiple="stack", color='blue', alpha=.5)
sns.kdeplot(data=hier_weight_mu_HC[0,1], ax=axs[1], multiple="stack", color='blue', alpha=.5)
sns.kdeplot(data=hier_weight_mu_HC[1,0], ax=axs[2], multiple="stack", color='blue', alpha=.5)
sns.kdeplot(data=hier_weight_mu_HC[1,1], ax=axs[3], multiple="stack", color='blue', alpha=.5)

fig.subplots_adjust(wspace=.3, hspace=.3)


# Check out if it does not exist
if not os.path.isdir(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/'):
        os.makedirs(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/') 


fig.savefig(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{model_name}_HC_individual_weighting.pdf')
plt.close()



# Extracting posterior distributions in PD
hier_weight_mu_PD = fit_PD["transfer_hier_weight_mu"] 
hier_alpha_pos_mu_PD = fit_PD["transfer_hier_alpha_pos_mu"]
hier_alpha_neg_mu_PD = fit_PD["transfer_hier_alpha_neg_mu"]
hier_sensitivity_mu_PD = fit_PD["transfer_hier_sensitivity_mu"] 

weight_PD = fit_PD["transfer_weight"] 
alpha_pos_PD = fit_PD["transfer_alpha_pos"]
alpha_neg_PD = fit_PD["transfer_alpha_neg"]
sensitivity_PD = fit_PD["transfer_sensitivity"] 

# Parkinson's disease

# Figure
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(30*mm, 30*mm))
axs = axs.flatten()

# action calaue learning
for i in range(weight_PD.shape[0]):
    # Act-Session 1
    sns.kdeplot(data=weight_PD[i,0,0], ax=axs[0], alpha=1)
    axs[0].set_xlim(0,1)
    axs[0].set_title('Act-OFF-PD, weighting parameter')
    #Outlier
    map_apex, dens_apex = MAP(weight_PD[i,0,0])
    if map_apex<.7:
        sub = participants_PD[i]
        axs[0].text(map_apex, dens_apex/20,  sub, transform=axs[0].transAxes, fontsize=8)

    # Act-Session 2
    sns.kdeplot(data=weight_PD[i,0,1], ax=axs[1], alpha=1)
    axs[1].set_xlim(0,1)
    axs[1].set_title('Act-ON-PD, weighting parameter')
    #Outlier
    map_apex, dens_apex = MAP(weight_PD[i,0,1])
    if map_apex<.7:
        sub = participants_PD[i]
        axs[1].text(map_apex, dens_apex/50,  sub, transform=axs[1].transAxes, fontsize=8)


    # Clr-Session 1
    sns.kdeplot(data=weight_PD[i,1,0], ax=axs[2], alpha=1)
    axs[2].set_xlim(0,1)
    axs[2].set_title('Clr-OFF-PD, weighting parameter')

    # Act-Session 2
    sns.kdeplot(data=weight_PD[i,1,1], ax=axs[3], alpha=1)
    axs[3].set_xlim(0,1)
    axs[3].set_title('Clr-ON-PD, weighting parameter')

# hierarchical parameter
sns.kdeplot(data=hier_weight_mu_PD[0,0], ax=axs[0], multiple="stack", color='blue', alpha=.5)
sns.kdeplot(data=hier_weight_mu_PD[0,1], ax=axs[1], multiple="stack", color='blue', alpha=.5)
sns.kdeplot(data=hier_weight_mu_PD[1,0], ax=axs[2], multiple="stack", color='blue', alpha=.5)
sns.kdeplot(data=hier_weight_mu_PD[1,1], ax=axs[3], multiple="stack", color='blue', alpha=.5)

fig.subplots_adjust(wspace=.3, hspace=.3)


# Check out if it does not exist
if not os.path.isdir(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/'):
        os.makedirs(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/') 


fig.savefig(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{model_name}_PD_individual_weighting.pdf')
plt.close() 


