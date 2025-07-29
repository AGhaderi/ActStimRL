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



# figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(21*cm, 5*cm))
axs = axs.flatten()


########################################################### Sensitivity in HC

sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[0,0], ax=axs[0], color='blue', alpha=.3, fill=True, linewidth=.1, label='Act-See1')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[0,1], ax=axs[0], color='blue', alpha=.5, fill=True, linewidth=.1, label='Act-See2')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[1,0], ax=axs[0], color='blue', alpha=.7, fill=True, linewidth=.1, label='Clr-See1')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC[1,1], ax=axs[0], color='blue', alpha=1, fill=True, linewidth=.1, label='Clr-See2')

axs[0].legend(fontsize=7, loc='upper left')
axs[0].set_xlim(0,.1)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_ylim(0,120)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Sensitivity parameter in HC', loc='left', fontsize=7)

# Bayes Factor
# session1
transfer_hier_sensitivity_mu_HC_sess1 = np.concatenate([transfer_hier_sensitivity_mu_HC[0,0], transfer_hier_sensitivity_mu_HC[1,0]])
# session2
transfer_hier_sensitivity_mu_HC_sess2 = np.concatenate([transfer_hier_sensitivity_mu_HC[0,1], transfer_hier_sensitivity_mu_HC[1,1]])
# Act
transfer_hier_sensitivity_mu_HC_Act = np.concatenate([transfer_hier_sensitivity_mu_HC[0,0], transfer_hier_sensitivity_mu_HC[0,1]])
# Clr
transfer_hier_sensitivity_mu_HC_Clr = np.concatenate([transfer_hier_sensitivity_mu_HC[1,0], transfer_hier_sensitivity_mu_HC[1,1]])

# session BF
i_sess = np.mean((transfer_hier_sensitivity_mu_HC_sess2 - transfer_hier_sensitivity_mu_HC_sess1)>0)
bf_sess = i_sess/(1-i_sess)
print(' Session effect in sensitivity in HC: ', bf_sess)
axs[0].text(.5, .9, f'BF(sess) = {round(bf_sess, 2)}', transform= axs[0].transAxes, fontsize=7)
# Condition BF
i_cond = np.mean((transfer_hier_sensitivity_mu_HC_Act - transfer_hier_sensitivity_mu_HC_Clr)>0)
bf_cond = i_cond/(1-i_cond)
print(' Condition effect in sensitivity in HC: ', bf_cond)
axs[0].text(.5, .8, f'BF(cond) = {round(bf_cond, 2)}', transform= axs[0].transAxes, fontsize=7)

 
 
########################################################### Valence sensitive learning rate in PD
 
sns.kdeplot(data=transfer_hier_sensitivity_mu_PD[0], ax=axs[1], color='red', alpha=.3, fill=True, linewidth=.1, label='OFF')
sns.kdeplot(data=transfer_hier_sensitivity_mu_PD[1], ax=axs[1], color='red', alpha=.7, fill=True, linewidth=.1, label='ON')

axs[1].legend(fontsize=7, loc='upper left')
axs[1].set_xlim(0,.1)
axs[1].tick_params(axis='both', labelsize=6)
axs[1].set_ylim(0,120)
axs[1].set_xlabel("", fontsize=6)
axs[1].set_ylabel("", fontsize=6)
axs[1].set_title('B) Sensitivity parameter in PD', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_sensitivity_mu_PD[1] - transfer_hier_sensitivity_mu_PD[0])>0)
bf = i/(1-i)
print(' Sensitivity parameter in PD: ', bf)
axs[1].text(.5, .9, f'BF = {round(bf, 2)}', transform= axs[1].transAxes, fontsize=7)


################################################################## Save image
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/{model_name}_HC_PD_sensitivity.png', dpi=500)


