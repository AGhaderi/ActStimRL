#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from utils import model_utils, config
import os

# set the state of random generator
rng = np.random.default_rng(321)

# name of model
model_name = 'tabel3_model1_complement_prob'
# The adrees name of pickle file
pickelDir_HC = f'{config.PROJECT_HIER_MODEL_DIR}/Tabel3/HC/{model_name}_HC.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{config.PROJECT_HIER_MODEL_DIR}/Tabel3/PD/{model_name}_PD.pkl'
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_HC = model_utils.load_pickle(load_path=pickelDir_HC)
loadPkl_PD = model_utils.load_pickle(load_path=pickelDir_PD)
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
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs[0], color=config.COLORS['PD-OFF'], fill=True, linewidth=1, alpha=.6,label='PD-OFF')
# HC session and session2 in Act
sns.kdeplot(data=np.mean([transfer_hier_weight_mu_HC[0,0], transfer_hier_weight_mu_HC[0,1]], axis=0), ax=axs[0], color=config.COLORS['HC'], fill=True, linewidth=1, alpha=.6,label='HC')
axs[0].legend(fontsize=6, loc='upper left')
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,80)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Disease related effect in AV condition', loc='left', fontsize=7)

# Bayes Factor
weight_HC_action = np.mean([transfer_hier_weight_mu_HC[0,0], transfer_hier_weight_mu_HC[0,1]], axis=0)

i = np.mean((weight_HC_action- transfer_hier_weight_mu_PD[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of action value learning betwen PD-OFF and HC: ', bf)
axs[0].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[0].transAxes, fontsize=7)

 
########################### Diease related effect (healthy control vs OFF state PD) in Color value learning

# PD OFF in Clr
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0], ax=axs[3], color=config.COLORS['PD-OFF'], fill=True, linewidth=1, alpha=.6,label='PD-OFF')
# HC session1 and session2 in Clr
sns.kdeplot(data=np.mean([transfer_hier_weight_mu_HC[1,0], transfer_hier_weight_mu_HC[1,1]], axis=0), ax=axs[3], color=config.COLORS['HC'], fill=True, linewidth=1, alpha=.6,label='HC')
axs[3].legend(fontsize=6, loc='upper left')
axs[3].set_xlim(0,1)
axs[3].tick_params(axis='both', labelsize=6)
axs[3].set_ylim(0,80)
axs[3].set_xlabel("", fontsize=6)
axs[3].set_ylabel("", fontsize=6)
axs[3].set_title('B) Disease related effect in CV condition', loc='left', fontsize=7)

# Bayes Factor
weight_HC_color = np.mean([transfer_hier_weight_mu_HC[1,0], transfer_hier_weight_mu_HC[1,1]], axis=0)
 
i = np.mean((weight_HC_color- transfer_hier_weight_mu_PD[1,0])>0)
bf = i/(1-i)
print('Weighting parameter of color value learning betwen PD-OFF and HC: ', bf)
axs[3].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[3].transAxes, fontsize=7)

########################################################### Medication effect in Parkinson's disease durting Action value Learning
# PD ON in Act
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,1], ax=axs[1], color=config.COLORS['PD-ON'], fill=True, linewidth=1, alpha=.6,label='PD-ON')
# PD OFF in Act
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs[1], color=config.COLORS['PD-OFF'], fill=True, linewidth=1, alpha=.6,label='PD-OFF')
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
# PD ON in Clr
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,1], ax=axs[4], color=config.COLORS['PD-ON'], fill=True, linewidth=1, alpha=.6,label='PD-ON')
# PD OFF in Clr
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0], ax=axs[4], color=config.COLORS['PD-OFF'], fill=True, linewidth=1, alpha=.6,label='PD-OFF')
axs[4].legend(fontsize=6, loc='upper left')
axs[4].set_xlim(0,1)
axs[4].tick_params(axis='both', labelsize=6)
axs[4].set_ylim(0,80)
axs[4].set_xlabel("", fontsize=6)
axs[4].set_ylabel("", fontsize=6)
axs[4].set_title('D) Medication effect in CV condition', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_weight_mu_PD[1,1]-transfer_hier_weight_mu_PD[1,0])>0)
bf = i/(1-i)
print('Weighting parameter of Color value learning across Medication in PD: ', bf)
axs[4].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[4].transAxes, fontsize=7)


########################################################### Session effect in Healthy control during Action value learning 
# HC session2 in Act
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,1], ax=axs[2], color=config.COLORS['HC-Sess2'], fill=True, linewidth=1, alpha=.6,label='HC-Sess2')
# HC session1 in Act
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,0], ax=axs[2], color=config.COLORS['HC-Sess1'], fill=True, linewidth=1, alpha=.6,label='HC-Sess1')
axs[2].legend(fontsize=6, loc='upper left')
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,80)
axs[2].tick_params(axis='both', labelsize=6)
axs[2].set_xlabel("", fontsize=6)
axs[2].set_ylabel("", fontsize=6)
axs[2].set_title('E) Repetition effect in AV condition', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_weight_mu_HC[0,1]-transfer_hier_weight_mu_HC[0,0])>0)
bf = i/(1-i)
print('Weighting parameter of action value learning across session in HC: ', bf)
axs[2].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[2].transAxes, fontsize=7)


############################## Session effect in Healthy control during Color value learning 
 # HC session2 in Clr
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,1], ax=axs[5], color=config.COLORS['HC-Sess2'], fill=True, linewidth=1, alpha=.6,label='HC-Sess2')
# HC session1 in Clr
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,0], ax=axs[5], color=config.COLORS['HC-Sess1'], fill=True, linewidth=1, alpha=.6,label='HC-Sess1')
axs[5].legend(fontsize=6, loc='upper left')
axs[5].set_xlim(0,1)
axs[5].tick_params(axis='both', labelsize=6)
axs[5].set_ylim(0,80)
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

# Check out if it does not exist
if not os.path.isdir(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/'):
        os.makedirs(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/') 


fig.savefig(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{model_name}_HC_PD_weighting.pdf')
plt.close()
