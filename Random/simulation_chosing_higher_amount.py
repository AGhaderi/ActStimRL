""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')

# Calculuate rewarded higher amount options
rawBehAll['pushed_agent'] = np.array(rawBehAll['winAmtPushable']>=50).astype(int)
rawBehAll['yellowChosen_agent'] = np.array(rawBehAll['winAmtYellow']>=50).astype(int)
# define rewarded or not rewarded option
rawBehAll['correctChoice_agent'] = (rawBehAll['block'] =='Act')*(rawBehAll['pushed_agent'] *rawBehAll['pushCorrect'] + (1-rawBehAll['pushed_agent'] )*(1-rawBehAll['pushCorrect'])) + (rawBehAll['block'] =='Stim')*(rawBehAll['yellowChosen_agent'] *rawBehAll['yellowCorrect'] + (1-rawBehAll['yellowChosen_agent'] )*(1-rawBehAll['yellowCorrect'])) 

# Save datafram as csv
parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation/agent'
# Check existing directory of subject name forlder and simulation number
if not os.path.isdir(f'{parent_dir}'):
    os.makedirs(f'{parent_dir}') 

rawBehAll.to_csv(f'{parent_dir}/high-amt-task-design-true-param.csv', index=False)
