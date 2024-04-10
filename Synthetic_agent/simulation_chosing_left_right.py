""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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
pushed_agent = rand*rawBehAll['leftCanBePushed'] + (1-rand)*(1-rawBehAll['leftCanBePushed'])
rawBehAll['pushed_agent'] = pushed_agent
# yellow chosen
yellowChosen_agent = rand*rawBehAll['yellowOnLeftSide'] + (1-rand)*(1-rawBehAll['yellowOnLeftSide'])
rawBehAll['yellowChosen_agent'] = yellowChosen_agent
# correct choice
correctChoice_agent = rand*rawBehAll['leftCorrect'] + (1-rand)*(1-rawBehAll['leftCorrect'])
rawBehAll['correctChoice_agent'] = correctChoice_agent

# Save datafram as csv
rawBehAll.to_csv('Simulation/simulation_chosing_left_right.csv', index=False)


# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']
