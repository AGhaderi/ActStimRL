#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from utils import *
import os

# name of model
model3_name = 'tabel1_model3'
model5_name = 'tabel1_model5'
# The address of model 3 and model 5 in HC
pickelDir_model3_HC = f'{SCRATCH_HIER_MODEL_DIR}/tabel1/model3/HC/{model3_name}_HC.pkl'
pickelDir_model5_HC = f'{SCRATCH_HIER_MODEL_DIR}/tabel1/model5/HC/{model5_name}_HC.pkl'
# The address of model 3 and model 5 in PD
pickelDir_model3_PD = f'{SCRATCH_HIER_MODEL_DIR}/tabel1/model3/PD/{model3_name}_PD.pkl'
pickelDir_model5_PD = f'{SCRATCH_HIER_MODEL_DIR}/tabel1/model5/PD/{model5_name}_PD.pkl'
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_model3_HC = load_pickle(load_path=pickelDir_model3_HC)
loadPkl_model5_HC = load_pickle(load_path=pickelDir_model5_HC)
loadPkl_model3_PD = load_pickle(load_path=pickelDir_model3_PD)
loadPkl_model5_PD = load_pickle(load_path=pickelDir_model5_PD)
fit_model3_HC = loadPkl_model3_HC['fit']
fit_model5_HC = loadPkl_model5_HC['fit']
fit_model3_PD = loadPkl_model3_PD['fit']
fit_model5_PD = loadPkl_model5_PD['fit']
 
# Extracting posterior distributions for each of four main unkhown parameters in HC
transfer_hier_weight_mu_model3_HC = fit_model3_HC["transfer_hier_weight_mu"] 
transfer_hier_weight_mu_model5_HC = fit_model5_HC["transfer_hier_weight_mu"] 
# Extracting posterior distributions for each of four main unkhown parameters in PD
transfer_hier_weight_mu_model3_PD = fit_model3_PD["transfer_hier_weight_mu"] 
transfer_hier_weight_mu_model5_PD = fit_model5_PD["transfer_hier_weight_mu"] 

# figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*cm, 10*cm))
axs = axs.flatten()


###########################################################  

# Color-value versus position-value learning in HC
sns.kdeplot(data=transfer_hier_weight_mu_model5_HC[0], ax=axs[0], color='grey', alpha=1, fill=True, linewidth=1, label='PV')
sns.kdeplot(data=transfer_hier_weight_mu_model5_HC[1], ax=axs[0], color='orange', alpha=1, fill=True, linewidth=1, label='CV')
axs[0].legend(fontsize=6, loc='upper left')
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,150)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Color-value versus position-value learning in HC', loc='left', fontsize=7)



# Color-value versus position-value learning in PD
sns.kdeplot(data=transfer_hier_weight_mu_model5_PD[0], ax=axs[1], color='grey', alpha=1, fill=True, linewidth=1, label='PV')
sns.kdeplot(data=transfer_hier_weight_mu_model5_PD[1], ax=axs[1], color='orange', alpha=1, fill=True, linewidth=1, label='CV')
axs[1].legend(fontsize=6, loc='upper left')
axs[1].set_xlim(0,1)
axs[1].set_ylim(0,150)
axs[1].tick_params(axis='both', labelsize=6)
axs[1].set_xlabel("", fontsize=6)
axs[1].set_ylabel("", fontsize=6)
axs[1].set_title('B) Color-value versus position-value learning in PD', loc='left', fontsize=7)



# Color-value versus action-value learning in HC
sns.kdeplot(data=transfer_hier_weight_mu_model3_HC[0], ax=axs[2], color='green', alpha=1, fill=True, linewidth=1, label='AV')
sns.kdeplot(data=transfer_hier_weight_mu_model3_HC[1], ax=axs[2], color='orange', alpha=1, fill=True, linewidth=1, label='CV')
axs[2].legend(fontsize=6, loc='upper left')
axs[2].set_xlim(0,1)
axs[2].set_ylim(0,115)
axs[2].tick_params(axis='both', labelsize=6)
axs[2].set_xlabel("", fontsize=6)
axs[2].set_ylabel("", fontsize=6)
axs[2].set_title('C) Color-value versus action-value learning in HC', loc='left', fontsize=7)

 
# Color-value versus action-value learning in PD
sns.kdeplot(data=transfer_hier_weight_mu_model3_PD[0], ax=axs[3], color='green', alpha=1, fill=True, linewidth=1, label='AV')
sns.kdeplot(data=transfer_hier_weight_mu_model3_PD[1], ax=axs[3], color='orange', alpha=1, fill=True, linewidth=1, label='CV')
axs[3].legend(fontsize=6, loc='upper left')
axs[3].set_xlim(0,1)
axs[3].set_ylim(0,115)
axs[3].tick_params(axis='both', labelsize=6)
axs[3].set_xlabel("", fontsize=6)
axs[3].set_ylabel("", fontsize=6)
axs[3].set_title('D) Color-value versus action-value learning in PD', loc='left', fontsize=7)

# Save image
plt.tight_layout()

# Check out if it does not exist
if not os.path.isdir(f'{SCRATCH_HIER_MODEL_DIR}/tabel1/'):
        os.makedirs(f'{SCRATCH_HIER_MODEL_DIR}/tabel1/') 

fig.savefig(f'{SCRATCH_HIER_MODEL_DIR}/tabel1/model1AND5AV_CV_OP_weighting.pdf')
