""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# read collected data across data
subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/BehData/'
# read collected data across all participants
behAll = pd.read_csv(f'{subMainDirec}/AllBehData/behAll.csv')
behAll = behAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})

"""This agent choose randomly left or right"""
 # Select random choice, left coded 1 and right coded 0
rand = np.random.binomial(1,.5, size = behAll.shape[0])
# pushed
behAll['pushed_agent'] = rand
behAll['wonAmount_agent'] = behAll['pushCorrect']*behAll['pushed_agent']*behAll['winAmtPushable'] + (1-behAll['pushCorrect'])*(1-behAll['pushed_agent'])*behAll['winAmtPullable']

# Save datafram as csv
parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation/agent'
# Check existing directory of subject name forlder and simulation number
if not os.path.isdir(f'{parent_dir}'):
    os.makedirs(f'{parent_dir}') 

behAll.to_csv(f'{parent_dir}/left-right-task-design-true-param.csv', index=False)
