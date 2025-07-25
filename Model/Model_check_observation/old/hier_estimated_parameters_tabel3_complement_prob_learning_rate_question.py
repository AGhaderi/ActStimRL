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

# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_alphaAct_pos_mu_PD = fit_PD["transfer_hier_alphaAct_pos_mu"] 
transfer_hier_alphaAct_neg_mu_PD = fit_PD["transfer_hier_alphaAct_neg_mu"] 
transfer_hier_alphaClr_pos_mu_PD = fit_PD["transfer_hier_alphaClr_pos_mu"] 
transfer_hier_alphaClr_neg_mu_PD = fit_PD["transfer_hier_alphaClr_neg_mu"] 
transfer_hier_weight_mu_PD = fit_PD["transfer_hier_weight_mu"] 
transfer_hier_sensitivity_mu_PD = fit_PD["transfer_hier_sensitivity_mu"]



# create a folder weighting_posterior
# Check out if it does not exist
if not os.path.isdir(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/'):
        os.makedirs(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/') 



########################################################### Posotive and negative learning rate across session/medication and condition in each group


############## Heirachcial parameter in Healthy control

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
# postive learning rate across condition and sessoin in HC
transfer_hier_alpha_pos_mu_HC = np.mean([transfer_hier_alphaAct_pos_mu_HC[0,0],transfer_hier_alphaAct_pos_mu_HC[1,0],
                                                transfer_hier_alphaClr_pos_mu_HC[0,1],transfer_hier_alphaClr_pos_mu_HC[1,1]], axis=0)
# Negative learning rate across condition and sessoin in HC
transfer_hier_alpha_neg_mu_HC = np.mean([transfer_hier_alphaAct_neg_mu_HC[0,0], transfer_hier_alphaAct_neg_mu_HC[1,0],
                                                transfer_hier_alphaClr_neg_mu_HC[0,1], transfer_hier_alphaClr_neg_mu_HC[1,1]], axis=0)
sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC, ax=axs, multiple="stack", color='blue', alpha=.8, label=r'$+ \alpha$')
sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC, ax=axs, multiple="stack", color='blue', alpha=.3, label=r'$-\alpha$')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,20)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_HC_pos_neg_learning_rate.png', dpi=500)


############################ Differnce in Parkinson's diease 

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7*mm, 4*mm))

sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC-transfer_hier_alpha_neg_mu_HC, ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-1,1)
axs.set_ylim(0,10)
#axs.set_xticks(np.arange(-1,1.3,.3))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_HC_pos_neg__Diff_learning_rate.png', dpi=500)
# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_HC - transfer_hier_alpha_neg_mu_HC)>0)
bf = i/(1-i)
print(' Positive-Negative Lernig rate in HC: ', bf)



############## Heirachcial parameter in Healthy control

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
# postive learning rate across condition and sessoin in PD
transfer_hier_alpha_pos_mu_PD = np.mean([transfer_hier_alphaAct_pos_mu_PD[0,0],transfer_hier_alphaAct_pos_mu_PD[1,0],
                                                transfer_hier_alphaClr_pos_mu_PD[0,1], transfer_hier_alphaClr_pos_mu_PD[1,1]], axis=0)
# negative learning rate across condition and sessoin in PD
transfer_hier_alpha_neg_mu_PD = np.mean([transfer_hier_alphaAct_neg_mu_PD[0,0],transfer_hier_alphaAct_neg_mu_PD[1,0],
                                                transfer_hier_alphaClr_neg_mu_PD[0,1], transfer_hier_alphaClr_neg_mu_PD[1,1]], axis=0)
sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD, ax=axs, multiple="stack", color='red', alpha=.8, label=r'$+ \alpha$')
sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD, ax=axs, multiple="stack", color='red', alpha=.3, label=r'$-\alpha$')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,20)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_PD_pos_neg_learning_rate.png', dpi=500)
 
############################ Difference in Parkinson's diease 

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7*mm, 4*mm))

sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD-transfer_hier_alpha_neg_mu_PD, ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-1,1)
axs.set_ylim(0,10)
#axs.set_xticks(np.arange(-1,1.3,.3))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_PD_pos_neg__Diff_learning_rate.png', dpi=500)

# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_PD - transfer_hier_alpha_neg_mu_PD)>0)
bf = i/(1-i)
print(' Positive-Negative Lernig rate in HC: ', bf)

######################################################################################




########################################################### Positive and negative learning rate across condition in each group


############## Heirachcial positive parameter in Healthy control

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
# postive learning rate across condition in HC
transfer_hier_alpha_pos_mu_HC_session1 = np.mean([transfer_hier_alphaAct_pos_mu_HC[0,0], transfer_hier_alphaClr_pos_mu_HC[0,1]], axis=0) #session 1
transfer_hier_alpha_pos_mu_HC_session2 = np.mean([transfer_hier_alphaAct_pos_mu_HC[1,0], transfer_hier_alphaClr_pos_mu_HC[1,1]], axis=0) #session 2

sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC_session1, ax=axs, multiple="stack", color='blue', alpha=.7, label='Sess1')
sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC_session2, ax=axs, multiple="stack", color='blue', alpha=.2, label='Sess2')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,20)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_HC_sesion_pos_learning_rate.png', dpi=500)

############## Difference of Heirachcial positive parameter in Healthy control

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7*mm, 4*mm))

sns.kdeplot(data=transfer_hier_alpha_pos_mu_HC_session2 - transfer_hier_alpha_pos_mu_HC_session1, ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-1,1)
axs.set_ylim(0,10)
#axs.set_xticks(np.arange(-1,1.3,.3))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_HC_sesion_Diff_pos_learning_rate.png', dpi=500)


# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_HC_session2 - transfer_hier_alpha_pos_mu_HC_session1)>0)
bf = i/(1-i)
print(' Positive Lernig rate across session in HC: ', bf)





############## Heirachcial negative parameter in Healthy control

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
# negative learning rate across condition in HC
transfer_hier_alpha_neg_mu_HC_session1 = np.mean([transfer_hier_alphaAct_neg_mu_HC[0,0], transfer_hier_alphaClr_neg_mu_HC[0,1]], axis=0) # session 1
transfer_hier_alpha_neg_mu_HC_session2 = np.mean([transfer_hier_alphaAct_neg_mu_HC[1,0], transfer_hier_alphaClr_neg_mu_HC[1,1]], axis=0) # session 2

sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC_session1, ax=axs, multiple="stack", color='blue', alpha=.7, label='Sess1')
sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC_session2, ax=axs, multiple="stack", color='blue', alpha=.2, label='Sess2')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,20)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_HC_sesion_neg_learning_rate.png', dpi=500)


############## Difference of Heirachcial negative parameter in Healthy control

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7*mm, 4*mm))

sns.kdeplot(data=transfer_hier_alpha_neg_mu_HC_session2 - transfer_hier_alpha_neg_mu_HC_session1, ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-1,1)
axs.set_ylim(0,10)
#axs.set_xticks(np.arange(-1,1.3,.3))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_HC_sesion_Diff_neg_learning_rate.png', dpi=500)

# Bayes Factor
i = np.mean((transfer_hier_alpha_neg_mu_HC_session2 - transfer_hier_alpha_neg_mu_HC_session1)>0)
bf = i/(1-i)
print(' Negative Lernig rate across session in HC: ', bf)




############## Heirachcial positive parameter in Parkinson's disease

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
# postive learning rate across condition in PD
transfer_hier_alpha_pos_mu_PD_OFF = np.mean([transfer_hier_alphaAct_pos_mu_PD[0,0], transfer_hier_alphaClr_pos_mu_PD[0,1]], axis=0) #OFF medication
transfer_hier_alpha_pos_mu_PD_ON = np.mean([transfer_hier_alphaAct_pos_mu_PD[1,0], transfer_hier_alphaClr_pos_mu_PD[1,1]], axis=0) #ON medication

sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD_OFF, ax=axs, multiple="stack", color='red', alpha=.2, label='PD-OFF')
sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD_ON, ax=axs, multiple="stack", color='red', alpha=.7, label='PD-ON')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,20)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_PD_medication_pos_learning_rate.png', dpi=500)

############## Difference of Heirachcial positive parameter in Parkinson's disease

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7*mm, 4*mm))

sns.kdeplot(data=transfer_hier_alpha_pos_mu_PD_ON - transfer_hier_alpha_pos_mu_PD_OFF, ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-1,1)
axs.set_ylim(0,10)
#axs.set_xticks(np.arange(-1,1.3,.3))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_PD_medication_Diff_pos_learning_rate.png', dpi=500)

# Bayes Factor
i = np.mean((transfer_hier_alpha_pos_mu_PD_ON - transfer_hier_alpha_pos_mu_PD_OFF)>0)
bf = i/(1-i)
print(' Positive Lernig rate across medication in PD: ', bf)
 


############## Heirachcial negative parameter in parkinsons's disease
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10*mm, 5*mm))
# negative learning rate across condition in PD
transfer_hier_alpha_neg_mu_PD_OFF = np.mean([transfer_hier_alphaAct_neg_mu_PD[0,0], transfer_hier_alphaClr_neg_mu_PD[0,1]], axis=0) # OFF medication
transfer_hier_alpha_neg_mu_PD_ON = np.mean([transfer_hier_alphaAct_neg_mu_PD[1,0], transfer_hier_alphaClr_neg_mu_PD[1,1]], axis=0) # ON medication

sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD_OFF, ax=axs, multiple="stack", color='red', alpha=.2, label='PD-OFF')
sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD_ON, ax=axs, multiple="stack", color='red', alpha=.7, label='PD-ON')
axs.legend(fontsize=6)
axs.set_xlim(0,1)
axs.set_ylim(0,20)
axs.set_xticks(np.arange(0,1.1,.1))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)
plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_PD_Medication_neg_learning_rate.png', dpi=500)


############## Difference of Heirachcial negative parameter in Parkinson's disease

mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7*mm, 4*mm))

sns.kdeplot(data=transfer_hier_alpha_neg_mu_PD_ON - transfer_hier_alpha_neg_mu_PD_OFF, ax=axs, multiple="stack", color='grey', alpha=.8)
axs.axvline(x = 0, color = 'green', linestyle='--')
#axs.set_xlim(-1,1)
axs.set_ylim(0,10)
#axs.set_xticks(np.arange(-1,1.3,.3))
axs.tick_params(axis='both', labelsize=6)
axs.set_xlabel("", fontsize=6)
axs.set_ylabel("Density", fontsize=6)

plt.tight_layout()
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/learning_rate_Posterior/{model_name}_PD_medication_Diff_neg_learning_rate.png', dpi=500)
 
# Bayes Factor
i = np.mean((transfer_hier_alpha_neg_mu_PD_ON - transfer_hier_alpha_neg_mu_PD_OFF)>0)
bf = i/(1-i)
print(' Negative Lernig rate across medication in PD: ', bf)
 





