#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
from Madule import plots
import arviz as az
from scipy import stats
from scipy.stats import gaussian_kde
import random
import os

# wirtten main directory  
writewriteMainScarch = '/mnt/scratch/projects/7TPD/amin'
# name of model
model_name = 'tabel3_model2_complement_prob'
# The adrees name of pickle file
pickelDir_HC = f'{writewriteMainScarch}/Behavioral/Tabel3/HC/tabel3_model1_complement_prob_HC.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{writewriteMainScarch}/Behavioral/Tabel3/PD/tabel3_model1_complement_prob_PD.pkl'
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_HC = utils.load_pickle(load_path=pickelDir_HC)
loadPkl_PD = utils.load_pickle(load_path=pickelDir_PD)
fit_HC = loadPkl_HC['fit']
fit_PD = loadPkl_PD['fit']
 
# Extracting posterior distributions for each of four main unkhown parameters in HC
transfer_hier_sensitivity_mu_HC = fit_HC["transfer_hier_sensitivity_mu"] 

# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_sensitivity_mu_PD = fit_PD["transfer_hier_sensitivity_mu"] 

# create a folder sensitivity_posterior
# Check out if it does not exist
if not os.path.isdir(f'{writewriteMainScarch}/Behavioral/Tabel3/sensitivity_Posterior/'):
        os.makedirs(f'{writewriteMainScarch}/Behavioral/Tabel3/sensitivity_Posterior/') 



########################################################### whether a sensitivity for HC

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))

sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[0,0], ax=axs, multiple="stack", color='blue', alpha=.1, label='Act-See1-')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[0,1], ax=axs, multiple="stack", color='blue', alpha=.4, label='Act-See2')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[1,0], ax=axs, multiple="stack", color='blue', alpha=.7, label='Clr-See1')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[1,1], ax=axs, multiple="stack", color='blue', alpha=1, label='Clr-See2')
axs.legend(fontsize=6, loc='upper left')
axs.set_ylim(0,120)
axs.set_xlim(0,.1)
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/sensitivity_Posterior/{model_name}_HC_sensitivy.png', dpi=500)


########################################################### whether a sensitivity for HC

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))

sns.kdeplot(data=transfer_hier_sensitivity_mu_PD[0], ax=axs, multiple="stack", color='red', alpha=.3, label='OFF')
sns.kdeplot(data=transfer_hier_sensitivity_mu_PD[1], ax=axs, multiple="stack", color='red', alpha=.8, label='ON')
axs.legend(fontsize=6, loc='upper left')
axs.set_ylim(0,120)
axs.set_xlim(0,.1)
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/sensitivity_Posterior/{model_name}_PD_sensitivy.png', dpi=500)

