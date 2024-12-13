""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})

# Calculuate rewarded left option
leftCorrect = rawBehAll['leftCanBePushed']*rawBehAll['pushCorrect'] + (1-rawBehAll['leftCanBePushed'])*(1-rawBehAll['pushCorrect'])
rawBehAll['leftCorrect'] = leftCorrect
 
"""This agent choose randomly left or right"""
 # Select random choice, left coded 1 and right coded 0
rand = np.random.binomial(1,.5, size = rawBehAll.shape[0])
# pushed
rawBehAll['pushed_agent'] = rand
# yellow chosen
yellowChosen_agent = rand*rawBehAll['yellowOnLeftSide'] + (1-rand)*(1-rawBehAll['yellowOnLeftSide'])
rawBehAll['yellowChosen_agent'] = yellowChosen_agent
# correct choice
correctChoice_agent = rand*rawBehAll['leftCorrect'] + (1-rand)*(1-rawBehAll['leftCorrect'])
rawBehAll['correctChoice_agent'] = correctChoice_agent

# Save datafram as csv
parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation/agent'
# Check existing directory of subject name forlder and simulation number
if not os.path.isdir(f'{parent_dir}'):
    os.makedirs(f'{parent_dir}') 

rawBehAll.to_csv(f'{parent_dir}/left-right-task-design-true-param.csv', index=False)
