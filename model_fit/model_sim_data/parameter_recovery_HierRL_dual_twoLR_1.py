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
import json 


# name of model
model_name = 'HierRL_dual_twoLR_paramer_recovery'

# number of simulation
n_simulations = 6
n_group = 2
n_session = 2
n_conds = 2
n_samples = 16000


# esimated parameters
estimated_hier_weight_mu = np.zeros((n_simulations, n_group, n_conds, n_samples))# n_sim, n_group, n_conds, n_samples
estimated_hier_alphaAct_pos_mu = np.zeros((n_simulations, n_group, n_samples)) # n_sim, n_group, n_samples
estimated_hier_alphaAct_neg_mu = np.zeros((n_simulations, n_group, n_samples)) 
estimated_hier_alphaClr_pos_mu = np.zeros((n_simulations, n_group, n_samples)) 
estimated_hier_alphaClr_neg_mu = np.zeros((n_simulations, n_group, n_samples)) 
estimated_hier_sensitivity_mu = np.zeros((n_simulations, n_group, n_samples))
estimated_hier_alpha_sd = np.zeros((n_simulations, n_group, n_samples))
estimated_hier_weight_sd = np.zeros((n_simulations, n_group, n_samples))
estimated_hier_sensitivity_sd = np.zeros((n_simulations, n_group, n_samples))

# grand truth
true_hier_weight_mu = np.zeros((n_simulations, n_group, n_conds))
true_hier_alphaAct_pos_mu = np.zeros((n_simulations, n_group)) # n_sim, n_cond,
true_hier_alphaAct_neg_mu = np.zeros((n_simulations, n_group))
true_hier_alphaClr_pos_mu = np.zeros((n_simulations, n_group))
true_hier_alphaClr_neg_mu = np.zeros((n_simulations, n_group))
true_hier_sensitivity_mu = np.zeros((n_simulations, n_group))
true_hier_alpha_sd = np.zeros((n_simulations, n_group))
true_hier_weight_sd = np.zeros((n_simulations, n_group, n_conds))
true_hier_sensitivity_sd = np.zeros((n_simulations, n_group))


# session effect over Parkinsdon's disease
for n in range(n_simulations):
    for group, partcipant_group in enumerate(['HC', 'PD']):
        # Directory of simulated data
        parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation'
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
        estimated_hier_alpha_sd[n, group] = fit['hier_alpha_sd'] 
        estimated_hier_weight_sd[n, group] = fit['hier_weight_sd'] 
        estimated_hier_sensitivity_sd[n, group] = fit['hier_sensitivity_sd'] 

for n in range(n_simulations):
    # grand truth
    with open(f'{parent_dir}/{str(n+1)}/hier-true-param.json', 'r') as openfile:
    
        # Reading from json file
        grand_trusth = json.load(openfile) 

    true_hier_weight_mu[n] = np.array(grand_trusth["hier_weight_mu"])
    true_hier_alphaAct_pos_mu[n] = np.array(grand_trusth["hier_alphaAct_pos_mu"])
    true_hier_alphaAct_neg_mu[n] = np.array(grand_trusth["hier_alphaAct_neg_mu"])
    true_hier_alphaClr_pos_mu[n] = np.array(grand_trusth["hier_alphaClr_pos_mu"])
    true_hier_alphaClr_neg_mu[n] = np.array(grand_trusth["hier_alphaClr_neg_mu"])
    true_hier_sensitivity_mu[n] =  np.array(grand_trusth["hier_sensitivity_mu"])
    true_hier_alpha_sd[n] = np.array(grand_trusth['hier_alpha_sd'])
    true_hier_weight_sd[n] = np.array(grand_trusth['hier_weight_sd'])
    true_hier_sensitivity_sd[n] = np.array(grand_trusth['hier_sensitivity_sd'])


############################################### figure of parameter recovery for some simulation
# plot of probability chosen left during trials
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(21*mm, 12.5*mm))
axs = axs.flatten()

idx = 1
for n in range(n_simulations): 
    for group in range(2):

            ### Positive Action Value learning
            # Compute percentiles for Healthy control group and Action condition
            bounds = stats.scoreatpercentile(estimated_hier_alphaAct_pos_mu[n,group,:], (.5, 2.5, 97.5, 99.5)) # sahpe in n_sim, n_group, n_samples
            grand_truth = true_hier_alphaAct_pos_mu[n, group] # # sahpe in n_sim, n_group
            axs[0].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[0].vlines(x = idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[0].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)


            ## Negative Action Value learning
            bounds = stats.scoreatpercentile(estimated_hier_alphaAct_neg_mu[n,group,:], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_alphaAct_neg_mu[n, group]
            axs[1].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[1].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[1].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
            
            ## Positive Color Value learning
            bounds = stats.scoreatpercentile(estimated_hier_alphaClr_pos_mu[n,group,:], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_alphaClr_pos_mu[n, group]
            axs[2].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[2].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[2].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)

            ## Negative Action Value learning
            bounds = stats.scoreatpercentile(estimated_hier_alphaClr_neg_mu[n,group,:], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_alphaClr_neg_mu[n, group]
            axs[3].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[3].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[3].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)

            ## Sensitiviy 
            bounds = stats.scoreatpercentile(estimated_hier_sensitivity_mu[n,group,:], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_sensitivity_mu[n, group]
            axs[4].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[4].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[4].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
        
            ## Weighting parameter for action value leanring
            bounds = stats.scoreatpercentile(estimated_hier_weight_mu[n,group,0, :], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_weight_mu[n, group,  0]
            axs[5].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[5].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[5].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)

            ## Weighting parameter for Color value leanring
            bounds = stats.scoreatpercentile(estimated_hier_weight_mu[n,group,1, :], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_weight_mu[n, group,  1]
            axs[6].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[6].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[6].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
       
            ## Weighting parameter SD
            bounds = stats.scoreatpercentile(estimated_hier_weight_sd[n,group,:], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_weight_sd[n, group, 0]
            axs[7].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[7].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[7].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)

            ## sensitivity SD
            bounds = stats.scoreatpercentile(estimated_hier_sensitivity_sd[n,group,:], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_sensitivity_sd[n, group]
            axs[8].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[8].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[8].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)

            ## learning rate SD
            bounds = stats.scoreatpercentile(estimated_hier_alpha_sd[n, group, :], (.5, 2.5, 97.5, 99.5))
            grand_truth = true_hier_alpha_sd[n, group]
            axs[9].vlines(x = idx, ymin =bounds[0], ymax=bounds[-1], color='deepskyblue', linewidth=1, alpha=.8)
            axs[9].vlines(x =idx, ymin =bounds[1], ymax=bounds[-2], color='blue', linewidth=5, alpha=.8)
            axs[9].plot(idx, grand_truth, marker='*', color= 'orange', alpha=1)
            idx +=1


### Positive Action Value learning
axs[0].set_ylabel(r'$+\mu^A_{(\alpha)}$', fontsize=10)
axs[0].set_xlabel('')
axs[0].set_ylim(-5,5)
axs[0].tick_params(bottom=False,top=False,labelbottom=False)

## Negative Action Value learning
axs[1].set_ylabel(r'$-\mu^A_{(\alpha)}$', fontsize=10)
axs[1].set_xlabel('')
axs[1].set_ylim(-5,5)
axs[1].tick_params(bottom=False,top=False,labelbottom=False)

## Positive Color Value learning
axs[2].set_ylabel(r'$+\mu^C_{(\alpha)}$', fontsize=10)
axs[2].set_xlabel('')
axs[2].set_ylim(-5,5)
axs[2].tick_params(bottom=False,top=False,labelbottom=False)

## Negative Action Value learning
axs[3].set_ylabel(r'$-\mu^C_{(\alpha)}$', fontsize=10)
axs[3].set_xlabel('')
axs[3].set_ylim(-5,5)
axs[3].tick_params(bottom=False,top=False,labelbottom=False)

## Sensitiviy 
axs[4].set_ylabel(r'$\mu_{(\beta)}$', fontsize=10)
axs[4].set_xlabel('')
axs[4].set_ylim(-8,0)
axs[4].tick_params(bottom=False,top=False,labelbottom=False)

## Weighting parameter for action value leanring
axs[5].set_ylabel(r'$\mu^A_{(w)}$', fontsize=10)
axs[5].set_xlabel('')
axs[5].set_ylim(-3,6)
axs[5].tick_params(bottom=False,top=False,labelbottom=False)

## Weighting parameter for Color value leanring
axs[6].set_ylabel(r'$\mu^C_{(w)}$', fontsize=10)
axs[6].set_xlabel('')
axs[6].set_ylim(-4,4)
axs[6].tick_params(bottom=False,top=False,labelbottom=False)


## Weighting parameter SD
axs[7].set_ylabel(r'$\sigma_{(w)}$', fontsize=10)
axs[7].set_xlabel('')
axs[7].set_ylim(-.5,4)
axs[7].tick_params(bottom=False,top=False,labelbottom=False)

## sensitivity SD
axs[8].set_ylabel(r'$\sigma_{(\beta)}$', fontsize=10)
axs[8].set_xlabel('')
axs[8].set_ylim(-.5,4)
axs[8].tick_params(bottom=False,top=False,labelbottom=False)

## learning rate SD
axs[9].set_ylabel(r'$\sigma_{(\alpha)}$', fontsize=10)
axs[9].set_xlabel('')
axs[9].set_ylim(-.5,4)
axs[9].tick_params(bottom=False,top=False,labelbottom=False)

axs[10].tick_params(bottom=False,top=False,labelbottom=False)
axs[11].tick_params(bottom=False,top=False,labelbottom=False)

# supper titile
fig.supxlabel('N simulation', fontsize= 12)
fig.suptitle('Hierarchical Parameter recovery', fontsize= 12)

## Adjust layout and show
plt.tight_layout()

## save
fig.savefig(f'{parent_dir}/{model_name}.png', dpi=500)

print('finish')