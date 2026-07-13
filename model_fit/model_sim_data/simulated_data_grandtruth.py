#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from utils import *
import os
from scipy import stats
import json

# grand truth of mean paramaters for each parameter
weight_Act = np.array([.4, .7,  1.0, 1.3, 1.6, 1.9, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6])
weight_Clr = np.array([-.4, -.7,  -1.0, -1.3, -1.6, -1.9, -2.1, -2.4, -2.7, -3.0, -3.3, -3.6])
learning_rate = np.array([-1.6, -1.3, -1, -.7, -.6, -.3, 0, .3, .6 , .9, 1.2, 1.5])
sensitivity = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.12])

# grand truth of sd paramaters for each parameter
alpha_sd = np.array([.01, .06, .11, .16, .21, .26, .31, .36, .41, .46, .51, .56])
weight_sd = np.array([.01, .06, .11, .16, .21, .26, .31, .36, .41, .46, .51, .56])
sensitivity_sd = np.array([.01, .06, .11, .16, .21, .26, .31, .36, .41, .46, .51, .56])

rng1 = np.random.default_rng(seed=1)
rng2 = np.random.default_rng(seed=2)
rng3 = np.random.default_rng(seed=3)
rng4 = np.random.default_rng(seed=4)

# Shuffle mean
weight_Act = rng1.choice(weight_Act, replace=False, size=12)
weight_Clr = rng1.choice(weight_Clr, replace=False, size=12)
sensitivity = rng1.choice(sensitivity, replace=False, size=12)
 
alphaAct_pos_mu = rng1.choice(learning_rate, replace=False, size=12)
alphaAct_neg_mu = rng2.choice(learning_rate, replace=False, size=12)
alphaClr_pos_mu = rng3.choice(learning_rate, replace=False, size=12)
alphaClr_neg_mu = rng4.choice(learning_rate, replace=False, size=12)

# Shuffle sd
alpha_sd = rng1.choice(alpha_sd, replace=False, size=12)
weight_sd = rng1.choice(weight_sd, replace=False, size=12)
sensitivity_sd = rng1.choice(sensitivity_sd, replace=False, size=12)

# Simulation number, from [0-11] resulting in 12 different simulations for each PD and HC group
sim = 0

# weighting parameter: [[[HC-Act, HC-Clr], [PD-Act, PD-Clr]]], , in shape [group, condition]
# learning rate and sensitivity: [[HC, PD]], in shape [group]
# mean
hier_weight_mu = [[weight_Act[2*(sim-1)], weight_Clr[2*(sim-1)]], [weight_Act[2*(sim-1)+1],weight_Clr[2*(sim-1)+1]]]
hier_alphaAct_pos_mu = [alphaAct_pos_mu[2*(sim-1)],alphaAct_pos_mu[2*(sim-1)+1]]
hier_alphaAct_neg_mu = [alphaAct_neg_mu[2*(sim-1)],alphaAct_neg_mu[2*(sim-1)+1]]
hier_alphaClr_pos_mu = [alphaClr_pos_mu[2*(sim-1)],alphaClr_pos_mu[2*(sim-1)+1]]
hier_alphaClr_neg_mu = [alphaClr_pos_mu[2*(sim-1)],alphaClr_neg_mu[2*(sim-1)+1]]
hier_sensitivity_mu = [sensitivity[2*(sim-1)], sensitivity[2*(sim-1)+1]]

# sd
hier_weight_sd = [[weight_sd[2*(sim-1)], weight_sd[2*(sim-1)]], [weight_sd[2*(sim-1)+1],weight_sd[2*(sim-1)+1]]]
hier_alpha_sd = [alpha_sd[2*(sim-1)], alpha_sd[2*(sim-1)+1]]
hier_sensitivity_sd = [sensitivity_sd[2*(sim-1)], sensitivity_sd[2*(sim-1)+1]]
 
# True values for each participant are randomly drown from predefined hierarchical level parameters, 
# generate data for each trials and individual will be saved in csv file
generating_hier_grand_truth(hier_weight_mu=hier_weight_mu, hier_alphaAct_pos_mu=hier_alphaAct_pos_mu, hier_alphaAct_neg_mu=hier_alphaAct_neg_mu,
                              hier_alphaClr_pos_mu=hier_alphaClr_pos_mu, hier_alphaClr_neg_mu=hier_alphaClr_neg_mu,
                              hier_sensitivity_mu=hier_sensitivity_mu, hier_alpha_sd=hier_alpha_sd, hier_weight_sd=hier_weight_sd, hier_sensitivity_sd=hier_sensitivity_sd,
                              sim=sim)

# simulate data from the grand truth parameters that has been generated from previous step
#simulate_hier_rl(sim=sim)

 