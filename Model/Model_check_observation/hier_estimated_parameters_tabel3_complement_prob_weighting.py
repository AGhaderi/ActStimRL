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
# set the state of random generator
rng = np.random.default_rng(321)

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
transfer_hier_weight_mu_HC = fit_HC["transfer_hier_weight_mu"] 

# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_weight_mu_PD = fit_PD["transfer_hier_weight_mu"] 

# figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(21*cm, 10*cm))
axs = axs.flatten()


########################################################### Diease related effect (healthy control vs OFF state PD) in Action value learning

# PD OFF in Act
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs[0], color='red', alpha=.3, fill=True, linewidth=.1, label='PD-OFF')
# HC session and session2 in Act
sns.kdeplot(data=np.concatenate([transfer_hier_weight_mu_HC[0,0], transfer_hier_weight_mu_HC[0,1]]), ax=axs[0], color='blue', alpha=.5, fill=True, linewidth=.1, label='HC')
axs[0].legend(fontsize=6, loc='upper left')
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,70)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Disease related effect in AV condition', loc='left', fontsize=7)

# Bayes Factor
weight_HC_action = np.concatenate([transfer_hier_weight_mu_HC[0,0], transfer_hier_weight_mu_HC[0,1]])
rng.shuffle(weight_HC_action)

i = np.mean((weight_HC_action[:24000]- transfer_hier_weight_mu_PD[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of action value learning betwen PD-OFF and HC: ', bf)
axs[0].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[0].transAxes, fontsize=7)

 
########################### Diease related effect (healthy control vs OFF state PD) in Color value learning

# PD OFF in Clr
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0], ax=axs[3], color='red', alpha=.3, fill=True, linewidth=.1, label='PD-OFF')
# HC session1 and session2 in Clr
sns.kdeplot(data=np.concatenate([transfer_hier_weight_mu_HC[1,0], transfer_hier_weight_mu_HC[1,1]]), ax=axs[3], color='blue', alpha=.5, fill=True, linewidth=.1, label='HC')
axs[3].legend(fontsize=6, loc='upper left')
axs[3].set_xlim(0,1)
axs[3].tick_params(axis='both', labelsize=6)
axs[3].set_ylim(0,30)
axs[3].set_xlabel("", fontsize=6)
axs[3].set_ylabel("", fontsize=6)
axs[3].set_title('B) Disease related effect in CV condition', loc='left', fontsize=7)

# Bayes Factor
weight_HC_color = np.concatenate([transfer_hier_weight_mu_HC[1,0], transfer_hier_weight_mu_HC[1,1]])
rng.shuffle(weight_HC_color)
 
i = np.mean((weight_HC_color[:24000]- transfer_hier_weight_mu_PD[1,0])>0)
bf = i/(1-i)
print('Weighting parameter of color value learning betwen PD-OFF and HC: ', bf)
axs[3].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[3].transAxes, fontsize=7)

########################################################### Medication effect in Parkinson's disease durting Action value Learning
# PD OFF in Act
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs[1], color='red', alpha=.3, fill=True, linewidth=.1, label='PD-OFF')
# PD ON in Act
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,1], ax=axs[1], color='red', alpha=.7, fill=True, linewidth=.1, label='PD-ON')
axs[1].legend(fontsize=6, loc='upper left')
axs[1].set_xlim(0,1)
axs[1].tick_params(axis='both', labelsize=6)
axs[1].set_ylim(0,30)
axs[1].set_xlabel("", fontsize=6)
axs[1].set_ylabel("", fontsize=6)
axs[1].set_title('C) Medication effect in AV condition', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_weight_mu_PD[0,1]-transfer_hier_weight_mu_PD[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of actoin value learning across Medication in PD: ', bf)
axs[1].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[1].transAxes, fontsize=7)

########################################################### Medication effect in Parkinson's disease durting Action value Learning
# PD OFF in Clr
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0], ax=axs[2], color='red', alpha=.3, fill=True, linewidth=.1, label='PD-OFF')
# PD ON in Clr
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,1], ax=axs[2], color='red', alpha=.7, fill=True, linewidth=.1, label='PD-ON')
axs[2].legend(fontsize=6, loc='upper left')
axs[2].set_xlim(0,1)
axs[2].tick_params(axis='both', labelsize=6)
axs[2].set_ylim(0,30)
axs[2].set_xlabel("", fontsize=6)
axs[2].set_ylabel("", fontsize=6)
axs[2].set_title('E) Medication effect in CV condition', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_weight_mu_PD[1,1]-transfer_hier_weight_mu_PD[1,0])>0)
bf = i/(1-i)
print('Weighting parameter of Color value learning across Medication in PD: ', bf)
axs[2].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[2].transAxes, fontsize=7)


########################################################### Session effect in Healthy control during Action value learning 
# HC session1 in Act
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,0], ax=axs[4], color='blue', alpha=.3, fill=True, linewidth=.1, label='HC-Sess1')
# HC session2 in Act
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,1], ax=axs[4], color='blue', alpha=.7, fill=True, linewidth=.1, label='HC-Sess2')
axs[4].legend(fontsize=6, loc='upper left')
axs[4].set_xlim(0,1)
axs[4].set_ylim(0,80)
axs[4].tick_params(axis='both', labelsize=6)
axs[4].set_xlabel("", fontsize=6)
axs[4].set_ylabel("", fontsize=6)
axs[4].set_title('D) Repetition effect in AV condition', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_weight_mu_HC[0,1]-transfer_hier_weight_mu_HC[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of action value learning across session in HC: ', bf)
axs[4].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[4].transAxes, fontsize=7)


############################## Session effect in Healthy control during Color value learning 
 # HC session1 in Clr
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,0], ax=axs[5], color='blue', alpha=.3, fill=True, linewidth=.1, label='HC-Sess1')
# HC session2 in Clr
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,1], ax=axs[5], color='blue', alpha=.7, fill=True, linewidth=.1, label='HC-Sess2')
axs[5].legend(fontsize=6, loc='upper left')
axs[5].set_xlim(0,1)
axs[5].tick_params(axis='both', labelsize=6)
axs[5].set_ylim(0,30)
axs[5].set_xlabel("", fontsize=6)
axs[5].set_ylabel("", fontsize=6)
axs[5].set_title('F) Repetition effect in CV condition', loc='left', fontsize=7)
# Bayes Factor
i = np.mean((transfer_hier_weight_mu_HC[1,1]-transfer_hier_weight_mu_HC[1,0])>0)
bf = i/(1-i)
print('Weighting parameter of color value learning across session in HC: ', bf)
axs[5].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[5].transAxes, fontsize=7)

# Save image
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/{model_name}_HC_PD_weighting.png', dpi=500)
