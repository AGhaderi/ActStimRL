""" Simulation study without RL model.
This agent takes the strategy of winstay and loseshift regardless of and regarding to Action and Color conditions in differet simulation"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
# Rename an column label
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})
 
"""This agent takes winStay/loseShift for pushing/pulling  in action condition and yellowChosen/BlueChosen Color conditions"""
choiceAgent = np.zeros(rawBehAll.shape[0])

for i in range(rawBehAll.shape[0]-1):
    # set the typs pf rewarded option regarding to block
    if rawBehAll['block'][i] =='Act':
        type = 'pushCorrect'
    elif rawBehAll['block'][i] =='Stim':
        type = 'yellowCorrect'
    # Check whether choice are 1 (push or yeloow) or 0 (pull or blue) and condition
    if (choiceAgent[i]==1) and (choiceAgent[i] == rawBehAll[type][i]):
        choiceAgent[i + 1] = 1 # win stay (push or yellow)
    elif (choiceAgent[i]==0) and (choiceAgent[i] == rawBehAll[type][i]):
        choiceAgent[i + 1] = 0 # win stay (pull or blue)
    elif (choiceAgent[i]==1) and (choiceAgent[i] != rawBehAll[type][i]):
        choiceAgent[i + 1] = 0  # lose shift (push or yellow)
    elif (choiceAgent[i]==0) and (choiceAgent[i] != rawBehAll[type][i]):
        choiceAgent[i + 1] = 1 # lose shift (pull or blue)

# Rewarded choice for agen
rawBehAll['correctAgent'] = (rawBehAll['block'] =='Act')*(choiceAgent*rawBehAll['pushCorrect'] + (1-choiceAgent)*(1-rawBehAll['pushCorrect'])) + (rawBehAll['block'] =='Stim')*(choiceAgent*rawBehAll['yellowCorrect'] + (1-choiceAgent)*(1-rawBehAll['yellowCorrect'])) 
# Wom amount for agen 
rawBehAll['wonAmtAgent'] = (rawBehAll['block'] =='Act')*(choiceAgent*rawBehAll['pushCorrect']*rawBehAll['winAmtPushable'] + (1-choiceAgent)*(1-rawBehAll['pushCorrect'])*rawBehAll['winAmtPullable']) + (rawBehAll['block'] =='Stim')*(choiceAgent*rawBehAll['yellowCorrect']*rawBehAll['winAmtYellow'] + (1-choiceAgent)*(1-rawBehAll['yellowCorrect'])*rawBehAll['winAmtBlue']) 

fig = plt.figure(figsize=(8, 4), tight_layout = True)

# total amount of observed data
behAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_wonAmount.loc[behAll_wonAmount['group']==2,'wonAmount'] = behAll_wonAmount[behAll_wonAmount['group']==2]['wonAmount']/2
sns.barplot(data = behAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")
# toral amount of agent 
behAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmtAgent'].sum()
behAll_wonAmount_agent.loc[behAll_wonAmount_agent['group']==2,'wonAmtAgent'] = behAll_wonAmount_agent[behAll_wonAmount_agent['group']==2]['wonAmtAgent']/2
sn = sns.barplot(data = behAll_wonAmount_agent, x='group', y='wonAmtAgent', hue='block', width=.5, errorbar="se", palette='Reds')
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Observed Act','Observed Clr', 'Agent 4 Act','Agent 4 Clr']):
    t.set_text(l)

plt.title('winStay/loseShift regarding Act and Clr conditions')
plt.ylabel('Total amount', fontsize=12)
plt.xlabel('Group label', fontsize=12)
plt.savefig('../figures/simulation4_total_ammount.png', dpi=300)
plt.show()


