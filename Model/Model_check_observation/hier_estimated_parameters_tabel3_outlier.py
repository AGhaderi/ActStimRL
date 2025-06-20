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

# wirtten main directory  
writewriteMainScarch = '/mnt/scratch/projects/7TPD/amin'
# name of model
model_name = 'tabel3_model1'
# The adrees name of pickle file
pickelDir_HC = f'{writewriteMainScarch}/Behavioral/Tabel3/HC/{model_name}_HC_cleanOutlier.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{writewriteMainScarch}/Behavioral/Tabel3/PD/{model_name}_PD_cleanOutlier.pkl'
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

########################################################### plot of learning rate in irrelavant condition group seperately
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(21*mm, 7*mm))
axs = axs.flatten()
fig.subplots_adjust(wspace=.3)

############ Healthy Control

# positive learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_HC[0,1], ax=axs[0], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_HC[1,1], ax=axs[0], multiple="stack", color='blue', alpha=.2)
axs[0].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[0].set_ylabel('HC')
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,7)

# negative learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_HC[0,1], ax=axs[1], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_HC[1,1], ax=axs[1], multiple="stack", color='blue', alpha=.2)
axs[1].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[1].set_ylabel('')
axs[1].set_xlim(0,1)

# positive learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_HC[0,0], ax=axs[2], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_HC[1,0], ax=axs[2], multiple="stack", color='blue', alpha=.2)
axs[2].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[2].set_ylabel('')
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,14)

# negatvie learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_HC[0,0], ax=axs[3], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_HC[1,0], ax=axs[3], multiple="stack", color='blue', alpha=.2)
axs[3].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[3].set_ylabel('')
axs[3].set_xlim(0,1)
axs[3].set_ylim(0,7)


############ Parkinson's disease

# positive learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_PD[0,1], ax=axs[4], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_PD[1,1], ax=axs[4], multiple="stack", color='red', alpha=.7)
axs[4].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[4].set_ylabel('PD')
axs[4].set_xlim(0,1)
axs[4].set_ylim(0,7)

# negative learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_PD[0,1], ax=axs[5], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_PD[1,1], ax=axs[5], multiple="stack", color='red', alpha=.7)
axs[5].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[5].set_ylabel('')
axs[5].set_xlim(0,1)

# positive learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_PD[0,0], ax=axs[6], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_PD[1,0], ax=axs[6], multiple="stack", color='red', alpha=.7)
axs[6].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[6].set_ylabel('')
axs[6].set_xlim(0,1)
axs[6].set_ylim(0,7)

# negatvie learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_PD[0,0], ax=axs[7], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_PD[1,0], ax=axs[7], multiple="stack", color='red', alpha=.7)
axs[7].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[7].set_ylabel('')
axs[7].set_xlim(0,1)
axs[7].set_ylim(0,7)

# Adjust layout and show
plt.tight_layout()
 
# save
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/{model_name}_irrelavant_lr_HC_cleanOutlier.png', dpi=500)


########################################################### plot of learning rate in relavant condition  group seperately
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(21*mm, 7*mm))
axs = axs.flatten()
fig.subplots_adjust(wspace=.3)

############ Healthy Control

# positive learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_HC[0,0], ax=axs[0], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_HC[1,0], ax=axs[0], multiple="stack", color='blue', alpha=.2)
i = np.mean((transfer_hier_alphaAct_pos_mu_HC[0,0] - transfer_hier_alphaAct_pos_mu_HC[1,0])>0)
bf = i/(1-i)
axs[0].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[0].transAxes, fontsize=8)
#axs[0].legend(['Ses1', 'Ses2'], fontsize=6)
axs[0].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[0].set_ylabel('HC', fontsize=10)
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,18)

# negative learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_HC[0,0], ax=axs[1], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_HC[1,0], ax=axs[1], multiple="stack", color='blue', alpha=.2)
i = np.mean((transfer_hier_alphaAct_neg_mu_HC[0,0] - transfer_hier_alphaAct_neg_mu_HC[1,0])>0)
bf = i/(1-i)
axs[1].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[1].transAxes, fontsize=8)
axs[1].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[1].set_ylabel('')
axs[1].set_xlim(0,1)
axs[1].set_ylim(0,7)

# positive learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_HC[0,1], ax=axs[2], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_HC[1,1], ax=axs[2], multiple="stack", color='blue', alpha=.2)
i = np.mean((transfer_hier_alphaClr_pos_mu_HC[0,1] - transfer_hier_alphaClr_pos_mu_HC[1,1])>0)
bf = i/(1-i)
axs[2].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[2].transAxes, fontsize=8)
axs[2].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[2].set_ylabel('')
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,7)
axs[2].set_ylim(0,7)

# negatvie learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_HC[0,1], ax=axs[3], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_HC[1,1], ax=axs[3], multiple="stack", color='blue', alpha=.2)
i = np.mean((transfer_hier_alphaClr_neg_mu_HC[0,1] - transfer_hier_alphaClr_neg_mu_HC[1,1])>0)
bf = i/(1-i)
axs[3].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[3].transAxes, fontsize=8)
axs[3].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[3].set_ylabel('')
axs[3].set_xlim(0,1)
axs[3].set_ylim(0,7)
axs[3].set_ylim(0,7)

############ Parkinson's disease

# positive learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_PD[0,0], ax=axs[4], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_PD[1,0], ax=axs[4], multiple="stack", color='red', alpha=.7)
i = np.mean((transfer_hier_alphaAct_pos_mu_PD[1,0] - transfer_hier_alphaAct_pos_mu_PD[0,0])>0)
bf = i/(1-i)
axs[4].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[4].transAxes, fontsize=8)
axs[4].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
#axs[4].legend(['OFF', 'ON'], fontsize=6)
axs[4].set_ylabel('PD', fontsize=10)
axs[4].set_xlim(0,1)
axs[4].set_ylim(0,7)
axs[4].set_ylim(0,15)

# negative learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_PD[0,0], ax=axs[5], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_PD[1,0], ax=axs[5], multiple="stack", color='red', alpha=.7)
i = np.mean((transfer_hier_alphaAct_neg_mu_PD[1,0] - transfer_hier_alphaAct_neg_mu_PD[0,0])>0)
bf = i/(1-i)
axs[5].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[5].transAxes, fontsize=8)
axs[5].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[5].set_ylabel('')
axs[5].set_xlim(0,1)
axs[5].set_ylim(0,7)

# positive learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_PD[0,1], ax=axs[6], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_PD[1,1], ax=axs[6], multiple="stack", color='red', alpha=.7)
i = np.mean((transfer_hier_alphaClr_pos_mu_PD[1,1] - transfer_hier_alphaClr_pos_mu_PD[0,1])>0)
bf = i/(1-i)
axs[6].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[6].transAxes, fontsize=8)
axs[6].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[6].set_ylabel('')
axs[6].set_xlim(0,1)
axs[6].set_ylim(0,7)
axs[6].set_ylim(0,7)

# negatvie learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_PD[0,1], ax=axs[7], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_PD[1,1], ax=axs[7], multiple="stack", color='red', alpha=.7)
i = np.mean((transfer_hier_alphaClr_neg_mu_PD[1,1] - transfer_hier_alphaClr_neg_mu_PD[0,1])>0)
bf = i/(1-i)
axs[7].text(.05, .80,  'bf={:.2f}'.format(bf), transform=axs[7].transAxes, fontsize=8)
axs[7].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[7].set_ylabel('')
axs[7].set_xlim(0,1)
axs[7].set_ylim(0,7)
axs[7].set_ylim(0,7)

# Adjust layout and show
plt.tight_layout()

# save
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/{model_name}_relavant_lr_HC_cleanOutlier.png', dpi=500)



########################################################### plot of weighting parameters each group seperately
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(21*mm, 5*mm))
axs = axs.flatten()
fig.subplots_adjust(wspace=.3)

############ Healthy Control

# weighting parameter  
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,0], ax=axs[0], multiple="stack", color='blue', alpha=.7, label='Ses1')
sns.kdeplot(data=transfer_hier_weight_mu_HC[0,1], ax=axs[0], multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,0], ax=axs[0], multiple="stack", color='blue', alpha=.2,  label='Ses2')
sns.kdeplot(data=transfer_hier_weight_mu_HC[1,1], ax=axs[0], multiple="stack", color='blue', alpha=.2)
axs[0].legend(fontsize=8, loc='upper left')
axs[0].set_xlabel(r'$\mu^A_{(w)}$', fontsize=10)
axs[0].set_ylabel('HC', fontsize=10)
axs[0].set_xlim(0,1)
# bf for color
i = np.mean((transfer_hier_weight_mu_HC[1,1] - transfer_hier_weight_mu_HC[0,1])>0)
bf_Clr = i/(1-i)
axs[0].text(.3, .80,  'bf-HC-Clr={:.2f}'.format(bf_Clr), transform=axs[0].transAxes, fontsize=8)
# bf for action
i = np.mean((transfer_hier_weight_mu_HC[1,0] - transfer_hier_weight_mu_HC[0,0])>0)
bf_Act = i/(1-i)
axs[0].text(.3, .70,  'bf-HC-Act={:.2f}'.format(bf_Act), transform=axs[0].transAxes, fontsize=8)



############ Parkinson's disease

# weighting parameter  

sns.kdeplot(data=transfer_hier_weight_mu_PD[0,0], ax=axs[1], multiple="stack", color='red', alpha=.2, label='OFF')
sns.kdeplot(data=transfer_hier_weight_mu_PD[0,1], ax=axs[1], multiple="stack", color='red', alpha=.2)
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,0], ax=axs[1], multiple="stack", color='red', alpha=.7, label='ON')
sns.kdeplot(data=transfer_hier_weight_mu_PD[1,1], ax=axs[1], multiple="stack", color='red', alpha=.7)
axs[1].legend(fontsize=8, loc='upper left')
axs[1].set_xlabel(r'$\mu^A_{(w)}$', fontsize=10)
axs[1].set_ylabel('PD', fontsize=10)
axs[1].set_xlim(0,1)
# bf for color
i = np.mean((transfer_hier_weight_mu_PD[1,1]-transfer_hier_weight_mu_PD[0,1])>0)
bf_Clr = i/(1-i)
axs[1].text(.3, .80,  'bf-PD-Clr={:.2f}'.format(bf_Clr), transform=axs[1].transAxes, fontsize=8)
# bf for action
i = np.mean((transfer_hier_weight_mu_PD[1,0]-transfer_hier_weight_mu_PD[0,0])>0)
bf_Act = i/(1-i)
axs[1].text(.3, .70,  'bf-PD-Act={:.2f}'.format(bf_Act), transform=axs[1].transAxes, fontsize=8)
 
 
 ############################# Disease effect
# bf for action
i = np.mean((transfer_hier_weight_mu_PD[0,0]-np.concatenate([transfer_hier_weight_mu_HC[0,0][random.sample(range(0,9000), 4500)], transfer_hier_weight_mu_HC[1,0][random.sample(range(0,9000), 4500)]]))>0)
bf_Act = (1-i)/i
axs[1].text(.3, .50,  'bf-Act={:.2f}'.format(bf_Act), transform=axs[1].transAxes, fontsize=8)
# bf for action
i = np.mean((transfer_hier_weight_mu_PD[0,1]-np.concatenate([transfer_hier_weight_mu_HC[0,1][random.sample(range(0,9000), 4500)], transfer_hier_weight_mu_HC[1,1][random.sample(range(0,9000), 4500)]]))>0)
bf_Act = (1-i)/i
axs[1].text(.3, .40,  'bf-Clr={:.2f}'.format(bf_Act), transform=axs[1].transAxes, fontsize=8)


# Adjust layout and show
plt.tight_layout()

# save
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel3/{model_name}_weighting_HC_cleanOutlier.png', dpi=500)

### 