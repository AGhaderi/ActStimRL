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


# wirtten main directory  
writewriteMainScarch = '/mnt/scratch/projects/7TPD/amin'
# name of model
model_name = 'tabel2_model1'
# The adrees name of pickle file
pickelDir_HC = f'{writewriteMainScarch}/Behavioral/Tabel2/HC/{model_name}_HC.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{writewriteMainScarch}/Behavioral/Tabel2/PD/{model_name}_PD.pkl'
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
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_HC[1], ax=axs[0], multiple="stack", color='blue', alpha=.7)
axs[0].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[0].set_ylabel('HC')
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,7)

# negative learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_HC[1], ax=axs[1], multiple="stack", color='blue', alpha=.7)
axs[1].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[1].set_ylabel('')
axs[1].set_xlim(0,1)

# positive learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_HC[0], ax=axs[2], multiple="stack", color='blue', alpha=.7)
axs[2].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[2].set_ylabel('')
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,14)

# negatvie learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_HC[0], ax=axs[3], multiple="stack", color='blue', alpha=.7)
axs[3].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[3].set_ylabel('')
axs[3].set_xlim(0,1)
axs[3].set_ylim(0,7)


############ Parkinson's disease

# positive learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_PD[1], ax=axs[4], multiple="stack", color='red', alpha=.7)
axs[4].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[4].set_ylabel('PD')
axs[4].set_xlim(0,1)
axs[4].set_ylim(0,7)

# negative learning rate of action value in Color condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_PD[1], ax=axs[5], multiple="stack", color='red', alpha=.7)
axs[5].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[5].set_ylabel('')
axs[5].set_xlim(0,1)

# positive learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_PD[0], ax=axs[6], multiple="stack", color='red', alpha=.7)
axs[6].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[6].set_ylabel('')
axs[6].set_xlim(0,1)
axs[6].set_ylim(0,7)

# negatvie learning rate of color value in Action condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_PD[0], ax=axs[7], multiple="stack", color='red', alpha=.7)
axs[7].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[7].set_ylabel('')
axs[7].set_xlim(0,1)
axs[7].set_ylim(0,7)

# Adjust layout and show
plt.tight_layout()
 
# save
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel2/{model_name}_irrelavant_lr.png', dpi=500)


########################################################### plot of learning rate in relavant condition  group seperately
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(21*mm, 7*mm))
axs = axs.flatten()
fig.subplots_adjust(wspace=.3)

############ Healthy Control

# positive learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_HC[0], ax=axs[0], multiple="stack", color='blue', alpha=.7)
axs[0].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[0].set_ylabel('HC', fontsize=10)
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,18)

# negative learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_HC[0], ax=axs[1], multiple="stack", color='blue', alpha=.7)
axs[1].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[1].set_ylabel('')
axs[1].set_xlim(0,1)
axs[1].set_ylim(0,7)

# positive learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_HC[1], ax=axs[2], multiple="stack", color='blue', alpha=.7)
axs[2].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[2].set_ylabel('')
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,7)
axs[2].set_ylim(0,7)

# negatvie learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_HC[1], ax=axs[3], multiple="stack", color='blue', alpha=.7)
axs[3].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[3].set_ylabel('')
axs[3].set_xlim(0,1)
axs[3].set_ylim(0,7)
axs[3].set_ylim(0,7)

############ Parkinson's disease

# positive learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_pos_mu_PD[0], ax=axs[4], multiple="stack", color='red', alpha=.7)
axs[4].set_xlabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
#axs[4].legend(['OFF', 'ON'], fontsize=6)
axs[4].set_ylabel('PD', fontsize=10)
axs[4].set_xlim(0,1)
axs[4].set_ylim(0,7)
axs[4].set_ylim(0,10)

# negative learning rate of action value in Action condition
sns.kdeplot(data=transfer_hier_alphaAct_neg_mu_PD[0], ax=axs[5], multiple="stack", color='red', alpha=.7)
axs[5].set_xlabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[5].set_ylabel('')
axs[5].set_xlim(0,1)
axs[5].set_ylim(0,7)

# positive learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_pos_mu_PD[1], ax=axs[6], multiple="stack", color='red', alpha=.7)
axs[6].set_xlabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[6].set_ylabel('')
axs[6].set_xlim(0,1)
axs[6].set_ylim(0,7)
axs[6].set_ylim(0,7)

# negatvie learning rate of color value in Color condition
sns.kdeplot(data=transfer_hier_alphaClr_neg_mu_PD[1], ax=axs[7], multiple="stack", color='red', alpha=.7)
axs[7].set_xlabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[7].set_ylabel('')
axs[7].set_xlim(0,1)
axs[7].set_ylim(0,7)
axs[7].set_ylim(0,7)

# Adjust layout and show
plt.tight_layout()

# save
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel2/{model_name}_relavant_lr.png', dpi=500)



########################################################### plot of weighting parameters each group seperately
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(21*mm, 10*mm))

############ Healthy Control

# weighting parameter  
sns.kdeplot(data=transfer_hier_weight_mu_HC[0], ax=axs, multiple="stack", color='blue', alpha=.7)
sns.kdeplot(data=transfer_hier_weight_mu_HC[1], ax=axs, multiple="stack", color='blue', alpha=.7)
axs.legend(fontsize=8, loc='upper left')
axs.set_xlabel(r'$\mu^A_{(w)}$', fontsize=10)
axs.set_xlim(0,1)


############ Parkinson's disease

# weighting parameter  

sns.kdeplot(data=transfer_hier_weight_mu_PD[0], ax=axs, multiple="stack", color='red', alpha=.7)
sns.kdeplot(data=transfer_hier_weight_mu_PD[1], ax=axs, multiple="stack", color='red', alpha=.7)
axs.legend(fontsize=8, loc='upper left')
axs.set_xlabel(r'$\mu^A_{(w)}$', fontsize=10)
axs.set_xlim(0,1)


# bf for color
i = np.mean((transfer_hier_weight_mu_PD[1] - transfer_hier_weight_mu_HC[1] )>0)
bf_Clr = i/(1-i)
axs.text(.3, .80,  'bf-Clr={:.2f}'.format(bf_Clr), transform=axs.transAxes, fontsize=14)
# bf for action
i = np.mean((transfer_hier_weight_mu_HC[0] - transfer_hier_weight_mu_PD[0])>0)
bf_Act = i/(1-i)
axs.text(.3, .70,  'bf-Act={:.2f}'.format(bf_Act), transform=axs.transAxes, fontsize=14)


# Adjust layout and show
plt.tight_layout()

# save
fig.savefig(f'{writewriteMainScarch}/Behavioral/Tabel2/{model_name}_weighting.png', dpi=500)

### 