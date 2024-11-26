""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
rawBehAll['group'] = rawBehAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])

# Rename an column label
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})
# Calculuate rewarded left option
leftCorrect = rawBehAll['leftCanBePushed']*rawBehAll.pushCorrect + (1-rawBehAll['leftCanBePushed'])*(1-rawBehAll.pushCorrect)
rawBehAll['leftCorrect'] = leftCorrect
 
# Select option with higher amount 
leftCorrect_higher_amt = rawBehAll['leftCorrect']*(rawBehAll['winAmtLeft']>=50)*rawBehAll['winAmtLeft']
rightCorrect_higher_amt  = (1-rawBehAll['leftCorrect'])*(rawBehAll['winAmtBlue']>=50)*rawBehAll['winAmtBlue']
agent_higher_ammount = leftCorrect_higher_amt + rightCorrect_higher_amt
rawBehAll['agent_higher_ammount'] = agent_higher_ammount 
# Select option with lower amount 
leftCorrect_lower_amt = rawBehAll['leftCorrect']*(rawBehAll['winAmtLeft']<50)*rawBehAll['winAmtLeft']
rightCorrect_lower_amt  = (1-rawBehAll['leftCorrect'])*(rawBehAll['winAmtBlue']<50)*rawBehAll['winAmtBlue']
agent_lower_ammount = leftCorrect_lower_amt + rightCorrect_lower_amt
rawBehAll['agent_lower_ammount'] = agent_lower_ammount

# figure
mm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(20*mm, 16*mm), tight_layout=True)
rows = 2
columns = 2

# Plot of left choice
fig.add_subplot(rows, columns, 1)
rawBehAll_high_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agent_higher_ammount'].sum()
rawBehAll_high_agent.loc[rawBehAll_high_agent['group']=='HC','agent_higher_ammount'] = rawBehAll_high_agent[rawBehAll_high_agent['group']=='HC']['agent_higher_ammount']/2
sns.barplot(data = rawBehAll_high_agent, x='group', y='agent_higher_ammount', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette='tab10')

rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sn = sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette ='pastel')
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act Agent', 'Clr Agent', 'Act Obs','Clr Obs']):
    t.set_text(l)
plt.setp(sn.get_legend().get_texts(), fontsize='8')  
plt.setp(sn.get_legend().get_title(), fontsize='8')  
plt.title('Choosing High-amount options')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12)
plt.ylim(0, 3600)

#Plot of right choice
fig.add_subplot(rows, columns, 2)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd",  palette ='pastel', legend=False)

rawBehAll_low_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agent_lower_ammount'].sum()
rawBehAll_low_agent.loc[rawBehAll_low_agent['group']=='HC','agent_lower_ammount'] = rawBehAll_low_agent[rawBehAll_low_agent['group']=='HC']['agent_lower_ammount']/2
sns.barplot(data = rawBehAll_low_agent, x='group', y='agent_lower_ammount', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette='tab10', legend=False)
plt.title('Choosing Low-amount option')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12)
plt.ylim(0, 3600)





"""This agent takes winStay/loseShift for pushing/pulling  in action condition and yellowChosen/BlueChosen Color conditions"""
fig.add_subplot(rows, columns, 3)


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


# total amount of observed data
behAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_wonAmount.loc[behAll_wonAmount['group']=='HC','wonAmount'] = behAll_wonAmount[behAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = behAll_wonAmount, x='group', y='wonAmount', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd",  palette ='pastel', legend=False)
# toral amount of agent 
behAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmtAgent'].sum()
behAll_wonAmount_agent.loc[behAll_wonAmount_agent['group']=='HC','wonAmtAgent'] = behAll_wonAmount_agent[behAll_wonAmount_agent['group']=='HC']['wonAmtAgent']/2
sns.barplot(data = behAll_wonAmount_agent, x='group', y='wonAmtAgent', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette='tab10', legend=False)
plt.title('winStay/loseShift')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12)
plt.ylim(0, 3600)



"""Agent chooses options with the higher probability of rewarding. """
# Plot of left choice
fig.add_subplot(rows, columns, 4)

# Rewarded choice for agen
rawBehAll['choiceAgent'] = np.nan

# List of subjects
subList = rawBehAll['sub_ID'].unique()

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

# toral amount of agent 6
rawBehAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmtAgent'].sum()
rawBehAll_wonAmount_agent.loc[rawBehAll_wonAmount_agent['group']=='HC','wonAmtAgent'] = rawBehAll_wonAmount_agent[rawBehAll_wonAmount_agent['group']=='HC']['wonAmtAgent']/2
sns.barplot(data = rawBehAll_wonAmount_agent, x='group', y='wonAmtAgent', hue='block',
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
     width=.5, errorbar="sd", palette='tab10', legend=False)
# total amount of observed data
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd",  palette ='pastel', legend=False)

plt.title('Choosing higher probability options')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12)
plt.ylim(0,3600)




# common x and y label
fig.supxlabel('Group label', fontsize='12')
fig.supylabel('Total amount', fontsize='12')
#fig.suptitle('', fontsize='12')

# save plot
plt.savefig('../Figures/simulation2_total_ammount.png', dpi=300)
plt.show()
 