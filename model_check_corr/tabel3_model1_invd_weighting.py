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

# set the state of random generator
rng = np.random.default_rng(321)
# name of model
model_name = 'model1'
# The adrees name of pickle file
pickelDir_HC = f'{SCRATCH_INDV_MODEL_DIR}/model1/HC/{model_name}_HC.pkl'
# pickle file in the scratch folder in PD
pickelDir_PD = f'{SCRATCH_INDV_MODEL_DIR}/model1/PD/{model_name}_PD.pkl'
"""Loading the pickle file of model fit from the subject directory"""
loadPkl_HC = load_pickle(load_path=pickelDir_HC)
loadPkl_PD = load_pickle(load_path=pickelDir_PD)
fit_HC = loadPkl_HC['fit']
fit_PD = loadPkl_PD['fit']
 
# Extracting posterior distributions for each of four main unkhown parameters in HC
weight_HC = fit_HC["weight"] 

# Extracting posterior distributions for each of four main unkhown parameters in PD
weight_PD = fit_PD["weight"] 

# figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(21*cm, 20*cm))
axs = axs.flatten()

# dimension of each group posteriors
nParts_HC, nConds_HC, nSess_HC, nSamples_HC = weight_HC.shape
nParts_PD, nConds_PD, nSess_PD, nSamples_PD = weight_PD.shape


########################################################### Diease related effect (healthy control vs OFF state PD) in Action value learning

# PD OFF in Act
for p_PD in range(nParts_PD):
        label= 'PD-OFF' if p_PD == 0 else None
        sns.kdeplot(data=weight_PD[p_PD,0,0], ax=axs[0], color=COLORS['PD-OFF'], fill=False, linewidth=1, alpha=1,label=label)

# HC session and session2 in Act
for p_HC in range(nParts_HC):
        label= 'HC' if p_HC == 0 else None
        weight_HC_action = np.mean([weight_HC[p_HC,0,0], weight_HC[p_HC,0,1]], axis=0)
        sns.kdeplot(data=weight_HC_action, ax=axs[0], color=COLORS['PD-OFF'], fill=False, linewidth=1, alpha=1,label=label)
axs[0].legend(fontsize=6, loc='upper left')
#axs[0].set_xlim(0,1)
#axs[0].set_ylim(0,90)
axs[0].tick_params(axis='both', labelsize=6)
axs[0].set_xlabel("", fontsize=6)
axs[0].set_ylabel("", fontsize=6)
axs[0].set_title('A) Disease related effect in AV condition', loc='left', fontsize=7)

 
########################### Diease related effect (healthy control vs OFF state PD) in Color value learning

# PD OFF in Clr
for p_PD in range(nParts_PD):
        label= 'PD-OFF' if p_PD == 0 else None
        sns.kdeplot(data=weight_PD[p_PD,1,0], ax=axs[1], color=COLORS['PD-OFF'], fill=False, linewidth=1, alpha=1,label=label)

# HC session1 and session2 in Clr
for p_HC in range(nParts_HC):
        label= 'HC' if p_HC == 0 else None
        weight_HC_color = np.mean([weight_HC[p_HC,1,0], weight_HC[p_HC,1,1]], axis=0)
        sns.kdeplot(data=weight_HC_color, ax=axs[1], color=COLORS['PD-OFF'], fill=False, linewidth=1, alpha=1,label=label)
axs[1].legend(fontsize=6, loc='upper left')
#axs[1].set_xlim(0,1)
axs[1].tick_params(axis='both', labelsize=6)
#axs[1].set_ylim(0,90)
axs[1].set_xlabel("", fontsize=6)
axs[1].set_ylabel("", fontsize=6)
axs[1].set_title('B) Disease related effect in CV condition', loc='left', fontsize=7)
 
 
########################################################### Medication effect in Parkinson's disease durting Action value Learning
# PD ON in Act
for p_PD in range(nParts_PD):
        label= 'PD-ON' if p_HC == 0 else None
        sns.kdeplot(data=weight_PD[p_PD,0,1], ax=axs[2], color=COLORS['PD-ON'], fill=False, linewidth=1, alpha=1,label=label)

# PD OFF in Act
for p_PD in range(nParts_PD):
        label= 'PD-OFF' if p_HC == 0 else None
        sns.kdeplot(data=weight_PD[p_PD,0,0], ax=axs[2], color=COLORS['PD-OFF'], fill=False, linewidth=1, alpha=1,label=label)
axs[2].legend(fontsize=6, loc='upper left')
#axs[2].set_xlim(0,1)
axs[2].tick_params(axis='both', labelsize=6)
#axs[2].set_ylim(0,40)
axs[2].set_xlabel("", fontsize=6)
axs[2].set_ylabel("", fontsize=6)
axs[2].set_title('C) Medication effect in AV condition', loc='left', fontsize=7)
 
########################################################### Medication effect in Parkinson's disease durting Action value Learning
# PD ON in Clr
for p_PD in range(nParts_PD):
        label= 'PD-ON' if p_PD == 0 else None
        sns.kdeplot(data=weight_PD[p_PD,1,1], ax=axs[3], color=COLORS['PD-ON'], fill=False, linewidth=1, alpha=1,label=label)

# PD OFF in Clr
for p_PD in range(nParts_PD):
        label= 'PD-OFF' if p_PD == 0 else None
        sns.kdeplot(data=weight_PD[p_PD,1,0], ax=axs[3], color=COLORS['PD-OFF'], fill=False, linewidth=1, alpha=1,label=label)
axs[3].legend(fontsize=6, loc='upper left')
#axs[3].set_xlim(0,1)
axs[3].tick_params(axis='both', labelsize=6)
#axs[3].set_ylim(0,90)
axs[3].set_xlabel("", fontsize=6)
axs[3].set_ylabel("", fontsize=6)
axs[3].set_title('D) Medication effect in CV condition', loc='left', fontsize=7)
 
########################################################### Session effect in Healthy control during Action value learning 
# HC session2 in Act
for p_HC in range(nParts_HC):
        label= 'HC-Sess2' if p_HC == 0 else None
        sns.kdeplot(data=weight_HC[p_HC,0,1], ax=axs[4], color=COLORS['HC-Sess2'], fill=False, linewidth=1, alpha=1,label=label)
# HC session1 in Act
for p_HC in range(nParts_HC):
        label= 'HC-Sess1' if p_HC == 0 else None
        sns.kdeplot(data=weight_HC[p_HC,0,0], ax=axs[4], color=COLORS['HC-Sess1'], fill=False, linewidth=1, alpha=1,label=label)
axs[4].legend(fontsize=6, loc='upper left')
#axs[4].set_xlim(0,1)
#axs[4].set_ylim(0,90)
axs[4].tick_params(axis='both', labelsize=6)
axs[4].set_xlabel("", fontsize=6)
axs[4].set_ylabel("", fontsize=6)
axs[4].set_title('E) Repetition effect in AV condition', loc='left', fontsize=7)

############################## Session effect in Healthy control during Color value learning 
 # HC session2 in Clr
for p_HC in range(nParts_HC):
        label= 'HC-Sess2' if p_HC == 0 else None
        sns.kdeplot(data=weight_HC[p_HC,1,1], ax=axs[5], color=COLORS['HC-Sess2'], fill=False, linewidth=1, alpha=1,label=label)
# HC session1 in Clr
for p_HC in range(nParts_HC):
        label= 'HC-Sess1' if p_HC == 0 else None
        sns.kdeplot(data=weight_HC[p_HC,1,0], ax=axs[5], color=COLORS['HC-Sess1'], fill=False, linewidth=1, alpha=1,label=label)
axs[5].legend(fontsize=6, loc='upper left')
#axs[5].set_xlim(0,1)
axs[5].tick_params(axis='both', labelsize=6)
#axs[5].set_ylim(0,90)
axs[5].set_xlabel("", fontsize=6)
axs[5].set_ylabel("", fontsize=6)
axs[5].set_title('F) Repetition effect in CV condition', loc='left', fontsize=7)

# Save image
#plt.tight_layout()

fig.savefig(f'{SCRATCH_INDV_MODEL_DIR}/{model_name}/{model_name}_HC_PD_weighting.pdf')
plt.close()



