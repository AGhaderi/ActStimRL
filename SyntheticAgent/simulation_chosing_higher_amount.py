""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')

# Calculuate rewarded higher amount options
rawBehAll['pushed_agent'] = np.array(rawBehAll['winAmtPushable']>=50).astype(int)
rawBehAll['yellowChosen_agent'] = np.array(rawBehAll['winAmtYellow']>=50).astype(int)
# define rewarded or not rewarded option
rawBehAll['correctChoice_agent'] = (rawBehAll['block'] =='Act')*(rawBehAll['pushed_agent'] *rawBehAll['pushCorrect'] + (1-rawBehAll['pushed_agent'] )*(1-rawBehAll['pushCorrect'])) + (rawBehAll['block'] =='Stim')*(rawBehAll['yellowChosen_agent'] *rawBehAll['yellowCorrect'] + (1-rawBehAll['yellowChosen_agent'] )*(1-rawBehAll['yellowCorrect'])) 

# Save datafram as csv
rawBehAll.to_csv('Simulation/simulation_chosing_higher_amount.csv', index=False)
