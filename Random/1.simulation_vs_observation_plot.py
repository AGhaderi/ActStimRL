""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/BehData/AllBehData/behAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
behAll['block'] = behAll['block'].replace(['Act', 'Stim'], ['Action', 'Color'])

# Rename an column label
behAll = behAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})
# Calculuate rewarded left option
leftCorrect = behAll['leftCanBePushed']*behAll.pushCorrect + (1-behAll['leftCanBePushed'])*(1-behAll.pushCorrect)
behAll['leftCorrect'] = leftCorrect
 
# Select option with higher amount 
leftCorrect_higher_amt = behAll['leftCorrect']*(behAll['winAmtLeft']>=50)*behAll['winAmtLeft']
rightCorrect_higher_amt  = (1-behAll['leftCorrect'])*(behAll['winAmtBlue']>=50)*behAll['winAmtBlue']
agent_higher_ammount = leftCorrect_higher_amt + rightCorrect_higher_amt
behAll['agent_higher_ammount'] = agent_higher_ammount 
# Select option with lower amount 
leftCorrect_lower_amt = behAll['leftCorrect']*(behAll['winAmtLeft']<50)*behAll['winAmtLeft']
rightCorrect_lower_amt  = (1-behAll['leftCorrect'])*(behAll['winAmtBlue']<50)*behAll['winAmtBlue']
agent_lower_ammount = leftCorrect_lower_amt + rightCorrect_lower_amt
behAll['agent_lower_ammount'] = agent_lower_ammount

# figure
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*mm, 16*mm), tight_layout=True)
axs = axs.flatten()

custom_palette = {
    'HC': 'blue',
    'PD-ON': 'red',
    'PD-OFF': '#FF7F7F'  # Light red in hex
}

# Plot of left choice
behAll_high_agent = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agent_higher_ammount'].sum()
behAll_high_agent.loc[behAll_high_agent['group']=='HC','agent_higher_ammount'] = behAll_high_agent[behAll_high_agent['group']=='HC']['agent_higher_ammount']/2
sn = sns.barplot(data = behAll_high_agent, x='block', y='agent_higher_ammount', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, ax=axs[0])

behAll_wonAmount = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_wonAmount.loc[behAll_wonAmount['group']=='HC','wonAmount'] = behAll_wonAmount[behAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = behAll_wonAmount, x='block', y='wonAmount', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, ax=axs[0])

for t, l in zip(sn.legend_.texts,['HC Agent', 'PD-ON Agent', 'PD-OFF Agent',
                                  'HC Obs', 'PD-ON Obs', 'PD-OFF Obs']):
    t.set_text(l)
sn.legend_.set_title('Group')
plt.setp(sn.get_legend().get_texts(), fontsize='6') # for legend text
plt.setp(sn.get_legend().get_title(), fontsize='6') # for legend title

axs[0].set_title('Choosing High-amount options')
axs[0].set_ylabel('', fontsize=12)
axs[0].set_xlabel('', fontsize=12)
axs[0].set_ylim(0, 3600)

###Choosing Low-amount option
behAll_wonAmount = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_wonAmount.loc[behAll_wonAmount['group']=='HC','wonAmount'] = behAll_wonAmount[behAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = behAll_wonAmount, x='block', y='wonAmount', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, legend=False, ax=axs[1])

behAll_low_agent = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agent_lower_ammount'].sum()
behAll_low_agent.loc[behAll_low_agent['group']=='HC','agent_lower_ammount'] = behAll_low_agent[behAll_low_agent['group']=='HC']['agent_lower_ammount']/2
sn1 = sns.barplot(data = behAll_low_agent, x='block', y='agent_lower_ammount', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, legend=False, ax=axs[1])
  
axs[1].set_title('Choosing Low-amount option')
axs[1].set_ylabel('', fontsize=12)
axs[1].set_xlabel('', fontsize=12)
axs[1].set_ylim(0, 3600)



"""This agent takes winStay/loseShift for pushing/pulling  in action condition and yellowChosen/BlueChosen Color conditions"""
choiceAgent = np.zeros(behAll.shape[0])

for i in range(behAll.shape[0]-1):
    # set the typs pf rewarded option regarding to block
    if behAll['block'][i] =='Action':
        type = 'pushCorrect'
    elif behAll['block'][i] =='Color':
        type = 'yellowCorrect'
    # Check whether choice are 1 (push or yeloow) or 0 (pull or blue) and condition
    if (choiceAgent[i]==1) and (choiceAgent[i] == behAll[type][i]):
        choiceAgent[i + 1] = 1 # win stay (push or yellow)
    elif (choiceAgent[i]==0) and (choiceAgent[i] == behAll[type][i]):
        choiceAgent[i + 1] = 0 # win stay (pull or blue)
    elif (choiceAgent[i]==1) and (choiceAgent[i] != behAll[type][i]):
        choiceAgent[i + 1] = 0  # lose shift (push or yellow)
    elif (choiceAgent[i]==0) and (choiceAgent[i] != behAll[type][i]):
        choiceAgent[i + 1] = 1 # lose shift (pull or blue)

# Rewarded choice for agen
behAll['correctAgent'] = (behAll['block'] =='Action')*(choiceAgent*behAll['pushCorrect'] + (1-choiceAgent)*(1-behAll['pushCorrect'])) + (behAll['block'] =='Color')*(choiceAgent*behAll['yellowCorrect'] + (1-choiceAgent)*(1-behAll['yellowCorrect'])) 
# Wom amount for agen 
behAll['wonAmtAgent'] = (behAll['block'] =='Action')*(choiceAgent*behAll['pushCorrect']*behAll['winAmtPushable'] + (1-choiceAgent)*(1-behAll['pushCorrect'])*behAll['winAmtPullable']) + (behAll['block'] =='Color')*(choiceAgent*behAll['yellowCorrect']*behAll['winAmtYellow'] + (1-choiceAgent)*(1-behAll['yellowCorrect'])*behAll['winAmtBlue']) 


# total amount of observed data
behAll_wonAmount = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_wonAmount.loc[behAll_wonAmount['group']=='HC','wonAmount'] = behAll_wonAmount[behAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = behAll_wonAmount, x='block', y='wonAmount', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, legend=False, ax = axs[2])

# toral amount of agent 
behAll_wonAmount_agent = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmtAgent'].sum()
behAll_wonAmount_agent.loc[behAll_wonAmount_agent['group']=='HC','wonAmtAgent'] = behAll_wonAmount_agent[behAll_wonAmount_agent['group']=='HC']['wonAmtAgent']/2
sn = sns.barplot(data = behAll_wonAmount_agent, x='block', y='wonAmtAgent', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, legend=False, ax = axs[2])

axs[2].set_title('winStay/loseShift')
axs[2].set_ylabel('', fontsize=12)
axs[2].set_xlabel('', fontsize=12)
axs[2].set_ylim(0, 3600)



"""Agent chooses options with the higher probability of rewarding. """

# Rewarded choice for agen
behAll['choiceAgent'] = np.nan

# List of subjects
subList = behAll['sub_ID'].unique()

for subName in subList:
    for sess in [1,2]:
        for run in [1,2]:
            for cond in ['Color', 'Action']:
                if cond=='Action':
                    # To select push and pull in action value learning
                    typeCorrect = 'pushCorrect'
                elif cond=='Color':
                    # To select yellow and blue in color value learning
                    typeCorrect = 'yellowCorrect'
                
                # take data from specific session, run for a subject
                actData = behAll[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)]
                # if the reversal point is 21
                if actData['reverse'].unique()==21:
                    # Phase 1
                    actDataPhase1 = actData[actData['phase']=='phase1']
                    propPhase1 = actDataPhase1[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']=='phase1'), 'choiceAgent'] = round(propPhase1)
                    # Phase 2
                    actDataPhase2 = actData[actData['phase']=='phase2']
                    propPhase2 = actDataPhase2[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']=='phase2'), 'choiceAgent'] = round(propPhase2)
                # if the reversal point is 14
                elif actData['reverse'].unique()==14:
                    # Phase 1
                    actDataPhase1 = actData[actData['phase']=='phase1']
                    propPhase1 = actDataPhase1[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']=='phase1'), 'choiceAgent'] = round(propPhase1)
                    # Phase 2
                    actDataPhase2 = actData[actData['phase']=='phase2']
                    propPhase2 = actDataPhase2[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']=='phase2'), 'choiceAgent'] = round(propPhase2)
                    # Phase 3
                    actDataPhase3 = actData[actData['phase']=='phase3']
                    propPhase3 = actDataPhase3[typeCorrect].mean()
                    # Chossing option with higher probability reward
                    behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']=='phase3'), 'choiceAgent'] = round(propPhase3)

# Wom amount for agen
behAll['wonAmtAgent'] = (behAll['block'] =='Action')*(behAll['choiceAgent'] *behAll['pushCorrect']*behAll['winAmtPushable'] + (1-behAll['choiceAgent'] )*(1-behAll['pushCorrect'])*behAll['winAmtPullable']) + (behAll['block'] =='Color')*(behAll['choiceAgent'] *behAll['yellowCorrect']*behAll['winAmtYellow'] + (1-behAll['choiceAgent'] )*(1-behAll['yellowCorrect'])*behAll['winAmtBlue']) 

# toral amount of agent 6
behAll_wonAmount_agent = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmtAgent'].sum()
behAll_wonAmount_agent.loc[behAll_wonAmount_agent['group']=='HC','wonAmtAgent'] = behAll_wonAmount_agent[behAll_wonAmount_agent['group']=='HC']['wonAmtAgent']/2
ax = sns.barplot(data = behAll_wonAmount_agent, x='block', y='wonAmtAgent', hue='group',
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
     width=.5, errorbar="sd", palette = custom_palette, legend=False, ax=axs[3])

# total amount of observed data
behAll_wonAmount = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_wonAmount.loc[behAll_wonAmount['group']=='HC','wonAmount'] = behAll_wonAmount[behAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = behAll_wonAmount, x='block', y='wonAmount', hue='group', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.2,
    alpha=1,
    width=.5, errorbar="sd", palette = custom_palette, legend=False, ax=axs[3])

axs[3].set_title('Choosing higher probability options')
axs[3].set_ylabel('', fontsize=12)
axs[3].set_xlabel('', fontsize=12)
axs[3].set_ylim(0,3600)




# common x and y label
fig.supxlabel('Group label', fontsize='12')
fig.supylabel('Total amount', fontsize='12')
#fig.suptitle('', fontsize='12')

# save plot
plt.savefig('../../Figures/simulation_vs_observation_plot.png', dpi=300)
  