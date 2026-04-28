#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from utils import model_utils, config
import arviz as az
from scipy import stats
from scipy.stats import gaussian_kde
import random
import os

# set the state of random generator
rng = np.random.default_rng(123)
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
transfer_hier_alpha_pos_mu_HC = fit_HC["transfer_hier_alpha_pos_mu"] 
transfer_hier_alpha_neg_mu_HC = fit_HC["transfer_hier_alpha_neg_mu"] 

# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_alpha_pos_mu_PD = fit_PD["transfer_hier_alpha_pos_mu"] 
transfer_hier_alpha_neg_mu_PD = fit_PD["transfer_hier_alpha_neg_mu"] 
 

# figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(21*cm, 10*cm))
axs = axs.flatten()


########################################################### Valence  sensitive learning rate in HC

# postive learning rate across sessoin in HC
transfer_hier_alpha_pos_mu_HC_sess = np.mean([transfer_hier_alpha_pos_mu_HC[0],transfer_hier_alpha_pos_mu_HC[1]], axis=0)
# Negative learning rate across sessoin in HC
transfer_hier_alpha_neg_mu_HC_sess = np.mean([transfer_hier_alpha_neg_mu_HC[0,0], transfer_hier_alpha_neg_mu_HC[0,1],
                                              transfer_hier_alpha_neg_mu_HC[1,0], transfer_hier_alpha_neg_mu_HC[1,1]], axis=0)

sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC_sess, ax=axs[3], color=config.COLORS['HC-NEG'], fill=True,linewidth=1, alpha=.6,  label=r'$- \alpha$')
sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC_sess, ax=axs[3], color=config.COLORS['HC-POS'], fill=True,linewidth=1, alpha=.6,  label=r'$+ \alpha$')

axs[3].legend(fontsize=6, loc='upper right')
axs[3].set_xlim(0,1)
axs[3].set_ylim(0,20)
axs[3].tick_params(axis='both', labelsize=6)
axs[3].set_xlabel("", fontsize=6)
axs[3].set_ylabel("", fontsize=6)
axs[3].set_title('B) Positive vs negative LR in HC', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_HC_sess - transfer_hier_alpha_neg_mu_HC_sess)>0)
bf = i/(1-i)
print(' Positive-Negative Lernig rate in HC: ', bf)
if bf>100:
        axs[3].text(.4, .9, f'BF > 100', transform= axs[3].transAxes, fontsize=7)
else:
        axs[3].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[3].transAxes, fontsize=7)

 
########################################################### Valence  sensitive learning rate in PD

# postive learning rate across medication in PD
transfer_hier_alpha_pos_mu_PD_med = np.mean([transfer_hier_alpha_pos_mu_PD[0],transfer_hier_alpha_pos_mu_PD[1]], axis=0)
# negative learning rate across medication in PD
transfer_hier_alpha_neg_mu_PD_med = np.mean([transfer_hier_alpha_neg_mu_PD[0,0],transfer_hier_alpha_neg_mu_PD[0,1],
                                             transfer_hier_alpha_neg_mu_PD[1,0],transfer_hier_alpha_neg_mu_PD[1,1]], axis=0)

sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD_med, ax=axs[0], color=config.COLORS['PD-NEG'], fill=True,linewidth=1, alpha=.6,  label=r'$- \alpha$')
sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD_med, ax=axs[0], color=config.COLORS['PD-POS'], fill=True,linewidth=1, alpha=.6,  label=r'$+ \alpha$')

axs[0].legend(fontsize=6, loc='upper right')
axs[0].set_xlim(0,1)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_ylim(0,20)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Positive vs negative LR in PD', loc='left', fontsize=7)

# Bayes Factor
# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_PD_med - transfer_hier_alpha_neg_mu_PD_med)>0)
bf = i/(1-i)
print(' Positive-Negative Lernig rate in HC: ', bf)
if bf>100:
        axs[0].text(.4, .9, f'BF > 100', transform= axs[0].transAxes, fontsize=7)
else:
        axs[0].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[0].transAxes, fontsize=7)

########################################################### Repetition effect of positive leraning rate in HC

# HC session 2
sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC[1], ax=axs[4], color=config.COLORS['HC-Sess2'], fill=True,linewidth=1, alpha=.6,  label='HC-sess2')
# HC session1
sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC[0], ax=axs[4], color=config.COLORS['HC-Sess1'], fill=True,linewidth=1, alpha=.6,  label='HC-sess1')
axs[4].legend(fontsize=6, loc='upper left')
axs[4].set_xlim(0,1)
axs[4].tick_params(axis='both', labelsize=6)
axs[4].set_ylim(0,20)
axs[4].set_xlabel("", fontsize=6)
axs[4].set_ylabel("", fontsize=6)
axs[4].set_title('D) Repetition effect in positive LR', loc='left', fontsize=7)

# Bayes Factor

# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_HC[1] - transfer_hier_alpha_pos_mu_HC[0])>0)
bf = i/(1-i)
print(' Positive Lernig rate across session in HC: ', bf)
axs[4].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[4].transAxes, fontsize=7)

########################################################### Repetition effect of negative leraning rate in HC
# HC session 2
transfer_hier_alpha_neg_mu_HC_sess2 = np.mean([transfer_hier_alpha_neg_mu_HC[0,1],transfer_hier_alpha_neg_mu_HC[1,1]], axis=0)
sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC_sess2, ax=axs[5], color=config.COLORS['HC-Sess2'], fill=True,linewidth=1, alpha=.6,  label='HC-sess2')

# HC session1
transfer_hier_alpha_neg_mu_HC_sess1 = np.mean([transfer_hier_alpha_neg_mu_HC[0,0],transfer_hier_alpha_neg_mu_HC[1,0]], axis=0)
sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC_sess1, ax=axs[5], color=config.COLORS['HC-Sess1'], fill=True,linewidth=1, alpha=.6,  label='HC-sess1')


axs[5].legend(fontsize=6, loc='upper left')
axs[5].set_xlim(0,1)
axs[5].tick_params(axis='both', labelsize=6)
axs[5].set_ylim(0,20)
axs[5].set_xlabel("", fontsize=6)
axs[5].set_ylabel("", fontsize=6)
axs[5].set_title('F) Repetition effect in negative LR', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_alpha_neg_mu_HC_sess2 - transfer_hier_alpha_neg_mu_HC_sess1)>0)
bf = i/(1-i)
print(' Negative Lernig rate across session in HC: ', bf)
axs[5].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[5].transAxes, fontsize=7)


########################################################### Medication effect of positive leraning rate 
# PD ON 
sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD[1], ax=axs[1], color=config.COLORS['PD-ON'], fill=True,linewidth=1, alpha=.6,  label='PD-ON')
# PD OFF
sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD[0], ax=axs[1], color=config.COLORS['PD-OFF'], fill=True,linewidth=1, alpha=.6,  label='PD-OFF')
axs[1].legend(fontsize=6, loc='upper left')
axs[1].set_xlim(0,1)
axs[1].set_ylim(0,20)
axs[1].tick_params(axis='both', labelsize=6)
axs[1].set_xlabel("", fontsize=6)
axs[1].set_ylabel("", fontsize=6)
axs[1].set_title('C) Medication effect in positive LR', loc='left', fontsize=7)

# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_PD[1] - transfer_hier_alpha_pos_mu_PD[0])>0)
bf = i/(1-i)
print(' Positive Lernig rate across medication in PD: ', bf)
axs[1].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[1].transAxes, fontsize=7)


##############################  Medication effect of negative leraning rate  

# PD ON 
transfer_hier_alpha_neg_mu_PD_sess2 = np.mean([transfer_hier_alpha_neg_mu_PD[0,1],transfer_hier_alpha_neg_mu_PD[1,1]], axis=0)
sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD_sess2, ax=axs[2], color=config.COLORS['PD-ON'], fill=True,linewidth=1, alpha=.6,  label='PD-ON')

# PD OFF
transfer_hier_alpha_neg_mu_PD_sess1 = np.mean([transfer_hier_alpha_neg_mu_PD[0,0],transfer_hier_alpha_neg_mu_PD[1,0]], axis=0)
sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD_sess1, ax=axs[2], color=config.COLORS['PD-OFF'], fill=True,linewidth=1, alpha=.6,  label='PD-OFF')

axs[2].legend(fontsize=6, loc='upper left')
axs[2].set_xlim(0,1)
axs[2].tick_params(axis='both', labelsize=6)
axs[2].set_ylim(0,20)
axs[2].set_xlabel("", fontsize=6)
axs[2].set_ylabel("", fontsize=6)
axs[2].set_title('E) Medication effect in negative LR', loc='left', fontsize=7)
# Bayes Factor
# Bayes Factor
i = np.mean((transfer_hier_alpha_neg_mu_PD_sess2 - transfer_hier_alpha_neg_mu_PD_sess1)>0)
bf = i/(1-i)
print(' Negative Lernig rate across medication in PD: ', bf)
axs[2].text(.4, .9, f'BF = {round(bf, 2)}', transform= axs[2].transAxes, fontsize=7)

# Save image
plt.tight_layout()

# Check out if it does not exist
if not os.path.isdir(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/'):
        os.makedirs(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/') 

fig.savefig(f'{config.SCRATCH_HIER_MODEL_DIR}/Tabel3/{model_name}_HC_PD_learning_rate.pdf')
plt.close()

 




