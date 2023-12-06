""" Simulation study without RL model.
Agent chooses options with the higher probability of rewarding. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
# Rename an column label
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})

# Rewarded choice for agen
rawBehAll['choiceAgent'] = np.nan

# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'csub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']

for subName in subList:
    for sess in [1,2]:
        for run in [1,2]:
            for cond in ['Stim', 'Act']:
                if cond=='Act':
                    # To select push and pull in action value learning
                    typeCorrect = 'pushCorrect'
                elif cond=='Stim':
                    # To select yellow and blue in color value learning
                    typeCorrect = 'yellowCorrect'
                
                # take data from specific session, run for a subject
                actData = rawBehAll[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)]
                # if the reversal point is 21
                if actData['reverse'].unique()==21:
                    # Phase 1
                    actDataPhase1 = actData[actData['phase']=='phase1']
                    propPhase1 = actDataPhase1[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase1'), 'choiceAgent'] = round(propPhase1)
                    # Phase 2
                    actDataPhase2 = actData[actData['phase']=='phase2']
                    propPhase2 = actDataPhase2[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase2'), 'choiceAgent'] = round(propPhase2)
                # if the reversal point is 14
                elif actData['reverse'].unique()==14:
                    # Phase 1
                    actDataPhase1 = actData[actData['phase']=='phase1']
                    propPhase1 = actDataPhase1[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase1'), 'choiceAgent'] = round(propPhase1)
                    # Phase 2
                    actDataPhase2 = actData[actData['phase']=='phase2']
                    propPhase2 = actDataPhase2[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase2'), 'choiceAgent'] = round(propPhase2)
                    # Phase 3
                    actDataPhase3 = actData[actData['phase']=='phase3']
                    propPhase3 = actDataPhase3[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    rawBehAll.loc[(rawBehAll['session']==sess) & (rawBehAll['run']==run) & (rawBehAll['block']==cond) & (rawBehAll['sub_ID']==subName)& (rawBehAll['phase']=='phase3'), 'choiceAgent'] = round(propPhase3)

# Wom amount for agen
rawBehAll['wonAmtAgent'] = (rawBehAll['block'] =='Act')*(rawBehAll['choiceAgent'] *rawBehAll['pushCorrect']*rawBehAll['winAmtPushable'] + (1-rawBehAll['choiceAgent'] )*(1-rawBehAll['pushCorrect'])*rawBehAll['winAmtPullable']) + (rawBehAll['block'] =='Stim')*(rawBehAll['choiceAgent'] *rawBehAll['yellowCorrect']*rawBehAll['winAmtYellow'] + (1-rawBehAll['choiceAgent'] )*(1-rawBehAll['yellowCorrect'])*rawBehAll['winAmtBlue']) 


fig = plt.figure(figsize=(8, 4), tight_layout = True)

# toral amount of agent 6
rawBehAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmtAgent'].sum()
rawBehAll_wonAmount_agent.loc[rawBehAll_wonAmount_agent['group']==2,'wonAmtAgent'] = rawBehAll_wonAmount_agent[rawBehAll_wonAmount_agent['group']==2]['wonAmtAgent']/2
sns.barplot(data = rawBehAll_wonAmount_agent, x='group', y='wonAmtAgent', hue='block', width=.5, errorbar="se", palette='Reds')
# total amount of observed data
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
sn = sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Observed Act','Observed Clr', 'Agent 5 Act','Agent 5 Clr']):
    t.set_text(l)

plt.title('winStay/loseShift regarding Act and Clr conditions')
plt.ylabel('Total amount', fontsize=12)
plt.xlabel('Group label', fontsize=12)
plt.ylim(0,3000)
plt.savefig('../figures/simulation5_total_ammount.png', dpi=300)
plt.show()


