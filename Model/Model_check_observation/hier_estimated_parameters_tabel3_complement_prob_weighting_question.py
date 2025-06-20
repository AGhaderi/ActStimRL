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
pickelDir_PD = f'{writewriteMainScarch}/Behavioral/Tabel3/PD/tabel3_model2_complement_prob_PD.pkl'
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_HC = utils.load_pickle(load_path=pickelDir_HC)
loadPkl_PD = utils.load_pickle(load_path=pickelDir_PD)
fit_HC = loadPkl_HC['fit']
fit_PD = loadPkl_PD['fit']
 
# Extracting posterior distributions for each of four main unkhown parameters in HC
transfer_hier_alphaAct_pos_mu_HC = fit_HC["transfer_hier_alphaAct_pos_mu"] 
transfer_hier_alphaAct_neg_mu_HC = fit_HC["transfer_hier_alphaAct_neg_mu"] 
transfer_hier_alphaClr_pos_mu_HC = fit_HC["transfer_hier_alphaClr_pos_mu"] 
transfer_hier_alphaClr_neg_mu_HC = fit_HC["transfer_hier_alphaClr_neg_mu"] 
transfer_hier_weight_mu_HC = fit_HC["transfer_hier_weight_mu"] 
transfer_hier_sensitivity_mu_HC = fit_HC["transfer_hier_sensitivity_mu"]

# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_alphaAct_pos_mu_PD = fit_PD["transfer_hier_alphaAct_pos_mu"] 
transfer_hier_alphaAct_neg_mu_PD = fit_PD["transfer_hier_alphaAct_neg_mu"] 
transfer_hier_alphaClr_pos_mu_PD = fit_PD["transfer_hier_alphaClr_pos_mu"] 
transfer_hier_alphaClr_neg_mu_PD = fit_PD["transfer_hier_alphaClr_neg_mu"] 
transfer_hier_weight_mu_PD = fit_PD["transfer_hier_weight_mu"] 
transfer_hier_sensitivity_mu_PD = fit_PD["transfer_hier_sensitivity_mu"]



# create a folder weighting_posterior
# Check out if it does not exist
if not os.path.isdir(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/'):
        os.makedirs(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/') 



########################################################### whether a weighting parameter PD<HC in Parkinson's disease

########################### weighting parameter for OFF and ON medication effect in Action value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))

sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs, multiple="stack", color='red', alpha=.2, label='PD-OFF')
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0], ax=axs, multiple="stack", color='red', alpha=.7, label='PD-ON')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,25)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_PD_Act_weighting.png', dpi=500)


############################ differnce of weighting parameter for OFF and ON medication effect in Action value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6*mm, 4*mm))

sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0]-transfer_hier_weight_mu_PD[0,0], ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-.5,.5)
#axs.set_xticks(np.arange(-.5,.7,.2))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_PD_Act_Diff_weighting.png', dpi=500)


# Bayes Factor
i = np.mean((transfer_hier_weight_mu_PD[1,0]-transfer_hier_weight_mu_PD[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of actoin value learning across Medication in PD: ', bf)


############################## weighting parameter for OFF and ON medication effect in Color value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
 
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,1], ax=axs, multiple="stack", color='red', alpha=.2, label='PD-OFF')
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,1], ax=axs, multiple="stack", color='red', alpha=.7, label='PD-ON')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,25)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_PD_Clr_weighting.png', dpi=500)


############################ differnce of weighting parameter for OFF and ON medication effect in Action value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6*mm, 4*mm))

sns.kdeplot(data=transfer_hier_weight_mu_PD[1,1]-transfer_hier_weight_mu_PD[0,1], ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-.5,.5)
#axs.set_xticks(np.arange(-.5,.7,.2))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_PD_Clr_Diff_weighting.png', dpi=500)


# Bayes Factor
i = np.mean((transfer_hier_weight_mu_PD[1,1]-transfer_hier_weight_mu_PD[0,1])>0)
bf = i/(1-i)
print('Weighting parameter of Color value learning across Medication in PD: ', bf)


########################################################### whether a weighting parameter session1<session2 in ParkinHealty Control

########################### weighting parameter for repetition effect in Action value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))

sns.kdeplot(data=transfer_hier_weight_mu_HC[0,0], ax=axs, multiple="stack", color='blue', alpha=.2, label='HC-Sess1')
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,0], ax=axs, multiple="stack", color='blue', alpha=.7, label='HC-Sess2')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_Act_weighting.png', dpi=500)

 
############################ differnce of weighting parameter for repetition effect in Action value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6*mm, 4*mm))

sns.kdeplot(data=transfer_hier_weight_mu_HC[1,0]-transfer_hier_weight_mu_HC[0,0], ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-.5,.5)
#axs.set_xticks(np.arange(-.5,.7,.2))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_Act_Diff_weighting.png', dpi=500)


# Bayes Factor
i = np.mean((transfer_hier_weight_mu_HC[1,0]-transfer_hier_weight_mu_HC[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of action value learning across session in HC: ', bf)


############################## weighting parameter for repetition effect in Color value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
 
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,1], ax=axs, multiple="stack", color='blue', alpha=.2, label='HC-Sess1')
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,1], ax=axs, multiple="stack", color='blue', alpha=.7, label='HC-Sess2')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,25)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_Clr_weighting.png', dpi=500)


############################ differnce of weighting parameter for repetition effect in Action value Learning
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6*mm, 4*mm))

sns.kdeplot(data=transfer_hier_weight_mu_HC[1,1]-transfer_hier_weight_mu_HC[0,1], ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-.5,.5)
#axs.set_xticks(np.arange(-.5,.7,.2))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_Clr_Diff_weighting.png', dpi=500)


# Bayes Factor
i = np.mean((transfer_hier_weight_mu_HC[1,1]-transfer_hier_weight_mu_HC[0,1])>0)
bf = i/(1-i)
print('Weighting parameter of color value learning across session in HC: ', bf)

########################################################### whether a weighting parameter has diease related effect (healthy control vs OFF state PD)

########################### weighting parameter for diease effect in Action value learning 
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
 
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs, multiple="stack", color='red', alpha=.2, label='PD-OFF')
sns.kdeplot(data=np.concatenate([transfer_hier_weight_mu_HC[0,0], transfer_hier_weight_mu_HC[1,0]]), ax=axs, multiple="stack", color='blue', alpha=.7, label='HC')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_PD_OFF_Act_weighting.png', dpi=500)
 

########################### differnce of weighting parameter for disease effect in Action value learning 
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6*mm, 4*mm))

weight_HC_action = np.concatenate([transfer_hier_weight_mu_HC[0,0], transfer_hier_weight_mu_HC[1,0]])
np.random.shuffle(weight_HC_action)
sns.kdeplot(data=weight_HC_action[:24000]- transfer_hier_weight_mu_PD[0,0], ax=axs, multiple="stack", color='grey', alpha=.8)

axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-.5,.5)
#axs.set_xticks(np.arange(-.5,.7,.2))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_PD_OFF_Act_Diff_weighting.png', dpi=500)


# Bayes Factor
i = np.mean((weight_HC_action[:24000]- transfer_hier_weight_mu_PD[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of action value learning betwen PD-OFF and HC: ', bf)

 
########################### weighting parameter for diease effect in Color value learning 
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))


sns.kdeplot(data=transfer_hier_weight_mu_PD[0,1], ax=axs, multiple="stack", color='red', alpha=.2, label='PD-OFF')
sns.kdeplot(data=np.concatenate([transfer_hier_weight_mu_HC[0,1], transfer_hier_weight_mu_HC[1,1]]), ax=axs, multiple="stack", color='blue', alpha=.7, label='HC')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,25)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_PD_OFF_Clr_weighting.png', dpi=500)
 

########################### differnce of weighting parameter for disease effect in Color value learning 
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6*mm, 4*mm))

weight_HC_color = np.concatenate([transfer_hier_weight_mu_HC[0,1], transfer_hier_weight_mu_HC[1,1]])
np.random.shuffle(weight_HC_color)
sns.kdeplot(data=weight_HC_color[:24000]- transfer_hier_weight_mu_PD[0,1], ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-.5,.5)
#axs.set_xticks(np.arange(-.5,.7,.2))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/weighting_Posterior/{model_name}_HC_PD_OFF_Clr_Diff_weighting.png', dpi=500)


# Bayes Factor
i = np.mean((weight_HC_color[:24000]- transfer_hier_weight_mu_PD[0,1])>0)
bf = i/(1-i)
print('Weighting parameter of color value learning betwen PD-OFF and HC: ', bf)

 