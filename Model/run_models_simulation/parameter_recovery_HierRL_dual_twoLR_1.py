#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

import numpy as np 
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
import nest_asyncio
import os
from scipy import stats

# name of model
model_name = 'HierRL_dual_twoLR_1'

# number of simulation
n_simulations = 3

# esimated parameters
estimated_hier_alphaAct_pos_mu = np.zeros((n_simulations,2, 2, 2, 16000)) # n_sim, n_group, n_session, n_cond, n_samples
estimated_hier_alphaAct_neg_mu = np.zeros((n_simulations,2, 2, 2, 16000)) 
estimated_hier_alphaClr_pos_mu = np.zeros((n_simulations,2, 2, 2, 16000)) 
estimated_hier_alphaClr_neg_mu = np.zeros((n_simulations,2, 2, 2, 16000)) 
estimated_hier_weight_mu = np.zeros((n_simulations,2, 2, 2, 16000))
estimated_hier_sensitivity_mu = np.zeros((n_simulations,2, 2, 2, 16000))
estimated_hier_alpha_sd = np.zeros((n_simulations, 2, 16000))
estimated_hier_weight_sd = np.zeros((n_simulations, 2, 16000))
estimated_hier_sensitivity_sd = np.zeros((n_simulations, 2, 16000))

# grand truth
true_hier_weight_mu = np.zeros((n_simulations, 2, 2))
true_hier_alphaAct_pos_mu = np.zeros((n_simulations, 2, 2)) # n_sim, n_session, n_cond,
true_hier_alphaAct_neg_mu = np.zeros((n_simulations, 2, 2))
true_hier_alphaClr_pos_mu = np.zeros((n_simulations, 2, 2))
true_hier_alphaClr_neg_mu = np.zeros((n_simulations, 2, 2))
true_hier_sensitivity_mu = np.zeros((n_simulations, 2, 2))
true_hier_alpha_sd = np.zeros(n_simulations)
true_hier_weight_sd = np.zeros(n_simulations)
true_hier_sensitivity_sd = np.zeros(n_simulations)


# session effect over Parkinsdon's disease
for n in range(n_simulations):
    for group, partcipant_group in enumerate(['HC']):
        # Directory of simulated data
        parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation/New Folder'
        # The adrees name of pickle file
        pickelDir = f'{parent_dir}/{str(n+1)}/{partcipant_group}/{model_name}.pkl'
        """Loading the pickle file of model fit from the subject directory if modelFit = False"""
        loadPkl = utils.load_pickle(load_path=pickelDir)
        fit = loadPkl['fit']

        # Extracting posterior distributions for each of four main unkhown parameters
        estimated_hier_weight_mu[n, group] = fit["hier_weight_mu"] 
        estimated_hier_alphaAct_pos_mu[n, group] = fit["hier_alphaAct_pos_mu"] 
        estimated_hier_alphaAct_neg_mu[n, group] = fit["hier_alphaAct_neg_mu"] 
        estimated_hier_alphaClr_pos_mu[n, group] = fit["hier_alphaClr_pos_mu"] 
        estimated_hier_alphaClr_neg_mu[n, group] = fit["hier_alphaClr_neg_mu"] 
        estimated_hier_sensitivity_mu[n, group] = fit["hier_sensitivity_mu"] 
        estimated_hier_alpha_sd[n, group] = fit['hier_alpha_sd'].flatten()
        estimated_hier_weight_sd[n, group] = fit['hier_weight_sd'].flatten()
        estimated_hier_sensitivity_sd[n, group] = fit['hier_sensitivity_sd'].flatten()

        # grand truth
        grand_trusth = pd.read_csv(f'{parent_dir}/{str(n+1)}/hier-true-param.csv')
        true_hier_weight_mu[n] = np.array(grand_trusth["hier_weight_mu"]).reshape(2,2).astype(int) 
        true_hier_alphaAct_pos_mu[n] = np.array(grand_trusth["hier_alphaAct_pos_mu"]).reshape(2,2).astype(int) 
        true_hier_alphaAct_neg_mu[n] = np.array(grand_trusth["hier_alphaAct_neg_mu"]).reshape(2,2).astype(int) 
        true_hier_alphaClr_pos_mu[n] = np.array(grand_trusth["hier_alphaClr_pos_mu"]).reshape(2,2).astype(int) 
        true_hier_alphaClr_neg_mu[n] = np.array(grand_trusth["hier_alphaClr_neg_mu"]).reshape(2,2).astype(int) 
        true_hier_sensitivity_mu[n] =  np.array(grand_trusth["hier_sensitivity_mu"]).reshape(2,2).astype(int)
        true_hier_alpha_sd[n] = np.array(grand_trusth['hier_alpha_sd'].unique()).astype(int)[0]
        true_hier_weight_sd[n] = np.array(grand_trusth['hier_weight_sd'].unique()).astype(int)[0]
        true_hier_sensitivity_sd[n] = np.array(grand_trusth['hier_sensitivity_sd'].unique()).astype(int)[0]

print(estimated_hier_alphaAct_pos_mu)


############################################### figure of parameter recovery for some simulation
# plot of probability chosen left during trials
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(20*mm, 20*mm))
axs = axs.flatten()
fig.subplots_adjust(wspace=.3)


### Positive Action Value learning
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_alphaAct_pos_mu[:,0,session,0], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_alphaAct_pos_mu[n, session, 0]
        axs[0].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[0].vlines(x = idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[0].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[0].set_ylabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[0].set_xlabel('')
axs[0].set_ylim(-2,6)
axs[0].tick_params(bottom=False,top=False,labelbottom=False)

# Negative Action Value learning
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_alphaAct_neg_mu[:,0,session,0], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_alphaAct_neg_mu[n, session, 0]
        axs[1].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[1].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[1].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[1].set_ylabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[1].set_xlabel('')
axs[1].set_ylim(-2,6)
axs[1].tick_params(bottom=False,top=False,labelbottom=False)

# Positive Color Value learning
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_alphaClr_pos_mu[:,0,session,1], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_alphaClr_pos_mu[n, session, 1]
        axs[2].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[2].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[2].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[2].set_ylabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[2].set_xlabel('')
axs[2].set_ylim(-2,6)
axs[2].tick_params(bottom=False,top=False,labelbottom=False)

# Negative Action Value learning
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_alphaClr_neg_mu[:,0,session,1], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_alphaClr_neg_mu[n, session, 1]
        axs[3].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[3].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[3].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[3].set_ylabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[3].set_xlabel('')
axs[3].set_ylim(-2,6)
axs[3].tick_params(bottom=False,top=False,labelbottom=False)

# Sensitiviy
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_sensitivity_mu[:,0,session,0], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_sensitivity_mu[n, session, 0]
        axs[4].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[4].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[4].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[4].set_ylabel(r'$\mu_{(\beta)}$', fontsize=10)
axs[4].set_xlabel('')
axs[4].set_ylim(-6,2)
axs[4].tick_params(bottom=False,top=False,labelbottom=False)

# Weighting parameter for action value leanring
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_weight_mu[:,0,session,0], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_weight_mu[n, session, 0]
        axs[5].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[5].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[5].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[5].set_ylabel(r'$\mu^A_{(w)}$', fontsize=10)
axs[5].set_xlabel('')
axs[5].set_ylim(-2,6)
axs[5].tick_params(bottom=False,top=False,labelbottom=False)

# Weighting parameter for Color value leanring
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_weight_mu[:,0,session,1], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_weight_mu[n, session, 1]
        axs[6].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[6].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[6].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[6].set_ylabel(r'$\mu^C_{(w)}$', fontsize=10)
axs[6].set_xlabel('')
axs[6].set_ylim(-6,2)
axs[6].tick_params(bottom=False,top=False,labelbottom=False)


# Weighting parameter SD
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_weight_sd[:,0,session], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_weight_sd[n]
        axs[7].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[7].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[7].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[7].set_ylabel(r'$\sigma_{(w)}$', fontsize=10)
axs[7].set_xlabel('')
axs[7].set_ylim(-.5,2)
axs[7].tick_params(bottom=False,top=False,labelbottom=False)

# sensitivity SD
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_sensitivity_sd[:,0,session], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_sensitivity_sd[n]
        axs[8].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[8].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[8].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[8].set_ylabel(r'$\sigma_{(\beta)}$', fontsize=10)
axs[8].set_xlabel('')
axs[8].set_ylim(-.5,2)
axs[8].tick_params(bottom=False,top=False,labelbottom=False)


# learning rate SD
idx = 1
for n in range(n_simulations): 
    for session in range(2):
        # Compute percentiles for Healthy control group and Action condition
        bounds = stats.scoreatpercentile(estimated_hier_alpha_sd[:,0,session], (.5, 2.5, 97.5, 99.5))
        grand_truth = true_hier_alpha_sd[n]
        axs[9].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
        axs[9].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
        axs[9].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        idx +=1
axs[9].set_ylabel(r'$\sigma_{(\alpha)}$', fontsize=10)
axs[9].set_xlabel('')
axs[9].set_ylim(-.5,2)
axs[9].tick_params(bottom=False,top=False,labelbottom=False)

axs[10].tick_params(bottom=False,top=False,labelbottom=False)
axs[11].tick_params(bottom=False,top=False,labelbottom=False)

# Adjust layout and show
plt.tight_layout()

#fig.supxlabel('N simulation', fontsize= 12)
#fig.suptitle('Hierarchical Parameter recovery', fontsize= 12)

# save
fig.savefig(f'{parent_dir}/{model_name}.png', dpi=500)
