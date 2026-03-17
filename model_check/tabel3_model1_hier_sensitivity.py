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

# wirtten main directory  
writewriteMainScarch = '/mnt/scratch/projects/7TPD/amin'
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
 
# Extracting posterior distributions for each of four main unkhown parameters in HC
transfer_hier_sensitivity_mu_HC = fit_HC["transfer_hier_sensitivity_mu"] 

# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_sensitivity_mu_PD = fit_PD["transfer_hier_sensitivity_mu"] 



# figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*cm, 10*cm))
axs = axs.flatten()


########################################################### Sensitivity in HC

# session
transfer_hier_sensitivity_mu_HC_sess1 = np.mean([transfer_hier_sensitivity_mu_HC[0,0], transfer_hier_sensitivity_mu_HC[1,0]], axis=0)
transfer_hier_sensitivity_mu_HC_sess2 = np.mean([transfer_hier_sensitivity_mu_HC[0,1], transfer_hier_sensitivity_mu_HC[1,1]], axis=0)

sns.kdeplot(data=transfer_hier_sensitivity_mu_HC_sess2, ax=axs[0], color=config.COLORS['HC-Sess2'], fill=True, linewidth=1, alpha=.6,  label='See2')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC_sess1, ax=axs[0], color=config.COLORS['HC-Sess1'], fill=True, linewidth=1, alpha=.6,  label='See1')

# condition
transfer_hier_sensitivity_mu_HC_Act = np.mean([transfer_hier_sensitivity_mu_HC[0,0], transfer_hier_sensitivity_mu_HC[0,1]], axis=0)
transfer_hier_sensitivity_mu_HC_Clr = np.mean([transfer_hier_sensitivity_mu_HC[1,0], transfer_hier_sensitivity_mu_HC[1,1]], axis=0)

sns.kdeplot(data=transfer_hier_sensitivity_mu_HC_Act, ax=axs[1], color=config.COLORS['HC-POS'], fill=True, linewidth=1, alpha=.6,  label='Act')
sns.kdeplot(data=transfer_hier_sensitivity_mu_HC_Clr, ax=axs[1], color=config.COLORS['HC-NEG'], fill=True, linewidth=1, alpha=.6,  label='Clr')

axs[0].legend(fontsize=7, loc='upper left')
axs[0].set_xlim(0.01,.08)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_ylim(0,130)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Repetition effect in sensitivity in HC', loc='left', fontsize=7)

axs[1].legend(fontsize=7, loc='upper left')
axs[1].set_xlim(0.01,.08)
axs[1].tick_params(axis='both', labelsize=6)
axs[1].set_ylim(0,130)
axs[1].set_xlabel("", fontsize=6)
axs[1].set_ylabel("", fontsize=6)
axs[1].set_title('B) Condition effect in sensitivity in HC', loc='left', fontsize=7)

# Bayes Factor
i_sess = np.mean((transfer_hier_sensitivity_mu_HC_sess2 - transfer_hier_sensitivity_mu_HC_sess1)>0)
bf_sess = i_sess/(1-i_sess)
print(' Session effect in sensitivity in HC: ', bf_sess)
axs[0].text(.5, .9, f'BF(sess) = {round(bf_sess, 2)}', transform= axs[0].transAxes, fontsize=7)
# Condition BF
i_cond = np.mean((transfer_hier_sensitivity_mu_HC_Act - transfer_hier_sensitivity_mu_HC_Clr)>0)
bf_cond = i_cond/(1-i_cond)
print(' Condition effect in sensitivity in HC: ', bf_cond)
axs[1].text(.5, .9, f'BF(cond) = {round(bf_cond, 2)}', transform= axs[1].transAxes, fontsize=7)

 
 
########################################################### Valence sensitive learning rate in PD
 

# Bayes Factor
# medication
transfer_hier_sensitivity_mu_PD_OFF = np.mean([transfer_hier_sensitivity_mu_PD[0,0], transfer_hier_sensitivity_mu_PD[1,0]], axis=0)
transfer_hier_sensitivity_mu_PD_ON = np.mean([transfer_hier_sensitivity_mu_PD[0,1], transfer_hier_sensitivity_mu_PD[1,1]], axis=0)

sns.kdeplot(data=transfer_hier_sensitivity_mu_PD_ON, ax=axs[2], color=config.COLORS['PD-ON'], fill=True, linewidth=1, alpha=.6,  label='ON')
sns.kdeplot(data=transfer_hier_sensitivity_mu_PD_OFF, ax=axs[2], color=config.COLORS['PD-OFF'], fill=True, linewidth=1, alpha=.6,  label='OFF')

# Condition
transfer_hier_sensitivity_mu_PD_Act = np.mean([transfer_hier_sensitivity_mu_PD[0,0], transfer_hier_sensitivity_mu_PD[0,1]], axis=0)
transfer_hier_sensitivity_mu_PD_Clr = np.mean([transfer_hier_sensitivity_mu_PD[1,0], transfer_hier_sensitivity_mu_PD[1,1]], axis=0)

sns.kdeplot(data=transfer_hier_sensitivity_mu_PD_Clr, ax=axs[3], color=config.COLORS['PD-NEG'], fill=True, linewidth=1, alpha=.6,  label='Clr')
sns.kdeplot(data=transfer_hier_sensitivity_mu_PD_Act, ax=axs[3], color=config.COLORS['PD-POS'], fill=True, linewidth=1, alpha=.6,  label='Act')

# Medication Effect BF
i_sess = np.mean((transfer_hier_sensitivity_mu_PD_OFF - transfer_hier_sensitivity_mu_PD_ON)>0)
bf_sess = i_sess/(1-i_sess)
print(' Medication effect in sensitivity in PD: ', bf_sess)
axs[2].text(.5, .9, f'BF(med) = {round(bf_sess, 2)}', transform= axs[2].transAxes, fontsize=7)
# Condition BF
i_cond = np.mean((transfer_hier_sensitivity_mu_PD_Act - transfer_hier_sensitivity_mu_PD_Clr)>0)
bf_cond = i_cond/(1-i_cond)
print(' Condition effect in sensitivity in PD: ', bf_cond)
axs[3].text(.5, .9, f'BF(cond) = {round(bf_cond, 2)}', transform= axs[3].transAxes, fontsize=7)



axs[2].legend(fontsize=7, loc='upper left')
axs[2].set_xlim(0.01,.08)
axs[2].tick_params(axis='both', labelsize=6)
axs[2].set_ylim(0,130)
axs[2].set_xlabel("", fontsize=6)
axs[2].set_ylabel("", fontsize=6)
axs[2].set_title('C) Medication effect in sensitivity in PD', loc='left', fontsize=7)

axs[3].legend(fontsize=7, loc='upper left')
axs[3].set_xlim(0.01,.08)
axs[3].tick_params(axis='both', labelsize=6)
axs[3].set_ylim(0,130)
axs[3].set_xlabel("", fontsize=6)
axs[3].set_ylabel("", fontsize=6)
axs[3].set_title('D) Condition effect in sensitivity in PD', loc='left', fontsize=7)

 
################################################################## Save image
plt.tight_layout()

# Check out if it does not exist
if not os.path.isdir(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/'):
        os.makedirs(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/') 
                       
fig.savefig(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{model_name}_HC_PD_sensitivity.pdf')
plt.close()


