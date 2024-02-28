""" Simulation study without RL model.
Agent chooses options with the higher probability of rewarding. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})

# Rewarded choice for agent
rawBehAll['pushed_agent'] = np.nan
rawBehAll['yellowChosen_agent'] = np.nan

# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']

for subName in subList:
    for sess in [1,2]:
        for run in [1,2]:
            for cond in ['Stim', 'Act']:
                if cond=='Act':
                    # take data from specific session, run for a subject
                    actData = rawBehAll[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)]
                    # if the reversal point is 21
                    if actData['reverse'].unique()==21:
                        # Phase 1
                        actDataPhase1 = actData[actData['phase']=='phase1']
                        propPhase1 = actDataPhase1['pushCorrect'].mean()
                        # Chossing option with higher probability reward, pushed
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase1'), 'pushed_agent'] = round(propPhase1)
                        # Phase 2
                        actDataPhase2 = actData[actData['phase']=='phase2']
                        propPhase2 = actDataPhase2['pushCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase2'), 'pushed_agent'] = round(propPhase2)
                    # if the reversal point is 14
                    elif actData['reverse'].unique()==14:
                        # Phase 1
                        actDataPhase1 = actData[actData['phase']=='phase1']
                        propPhase1 = actDataPhase1['pushCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase1'), 'pushed_agent'] = round(propPhase1)
                        # Phase 2
                        actDataPhase2 = actData[actData['phase']=='phase2']
                        propPhase2 = actDataPhase2['pushCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase2'), 'pushed_agent'] = round(propPhase2)
                        # Phase 3
                        actDataPhase3 = actData[actData['phase']=='phase3']
                        propPhase3 = actDataPhase3['pushCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase3'), 'pushed_agent'] = round(propPhase3)

                elif cond=='Stim':
                    # take data from specific session, run for a subject
                    actData = rawBehAll[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)]
                    # if the reversal point is 21
                    if actData['reverse'].unique()==21:
                        # Phase 1
                        actDataPhase1 = actData[actData['phase']=='phase1']
                        propPhase1 = actDataPhase1['yellowCorrect'].mean()
                        # Chossing option with higher probability reward, pushed
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase1'), 'yellowChosen_agent'] = round(propPhase1)
                        # Phase 2
                        actDataPhase2 = actData[actData['phase']=='phase2']
                        propPhase2 = actDataPhase2['yellowCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase2'), 'yellowChosen_agent'] = round(propPhase2)
                    # if the reversal point is 14
                    elif actData['reverse'].unique()==14:
                        # Phase 1
                        actDataPhase1 = actData[actData['phase']=='phase1']
                        propPhase1 = actDataPhase1['yellowCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase1'), 'yellowChosen_agent'] = round(propPhase1)
                        # Phase 2
                        actDataPhase2 = actData[actData['phase']=='phase2']
                        propPhase2 = actDataPhase2['yellowCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase2'), 'yellowChosen_agent'] = round(propPhase2)
                        # Phase 3
                        actDataPhase3 = actData[actData['phase']=='phase3']
                        propPhase3 = actDataPhase3['yellowCorrect'].mean()
                        # Chossing option with higher probability reward
                        rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase3'), 'yellowChosen_agent'] = round(propPhase3)

# the corresponging yellow option in action value condition
rawBehAll_Act = rawBehAll[(rawBehAll['block']=='Act')]
pushed_agent = np.array(rawBehAll_Act['pushed_agent'])
isEqual_push_yellow = np.array(rawBehAll_Act['yellowOnLeftSide']==rawBehAll_Act['leftCanBePushed'])
yellowChosen_agent = np.zeros(len(rawBehAll_Act))

for i in range(len(pushed_agent)):
    if isEqual_push_yellow[i]==True:
        yellowChosen_agent[i] = pushed_agent[i]
    elif isEqual_push_yellow[i]==False:
        yellowChosen_agent[i] = 1-pushed_agent[i]
rawBehAll.loc[rawBehAll['block']=='Act', 'yellowChosen_agent'] = yellowChosen_agent

# the corresponging yellow option in color value condition
rawBehAll_Stim = rawBehAll[(rawBehAll['block']=='Stim')]
yellowChosen_agent = np.array(rawBehAll_Stim['yellowChosen_agent'])
isEqual_push_yellow = np.array(rawBehAll_Stim['yellowOnLeftSide']==rawBehAll_Stim['leftCanBePushed'])
pushed_agent = np.zeros(len(rawBehAll_Stim))
for i in range(len(rawBehAll_Stim)):
    if isEqual_push_yellow[i]==True:
        pushed_agent[i]= yellowChosen_agent[i]
    elif isEqual_push_yellow[i]==False:
        pushed_agent[i]= 1-yellowChosen_agent[i]
rawBehAll.loc[rawBehAll['block']=='Stim', 'pushed_agent'] = pushed_agent

# define correct choice
rawBehAll['correctChoice_agent'] = (rawBehAll['block'] =='Act')*(rawBehAll['pushed_agent'] *rawBehAll['pushCorrect'] + (1-rawBehAll['pushed_agent'] )*(1-rawBehAll['pushCorrect'])) + (rawBehAll['block'] =='Stim')*(rawBehAll['yellowChosen_agent'] *rawBehAll['yellowCorrect'] + (1-rawBehAll['yellowChosen_agent'] )*(1-rawBehAll['yellowCorrect'])) 

# Save datafram as csv
rawBehAll.to_csv('Simulation/simulation_chosing_higher_probability.csv', index=False)
 