""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Behavioral_modeling/BehData/AllBehData/behAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
behAll['block'] = behAll['block'].replace(['Act', 'Stim'], ['Action', 'Color'])
# Rename an column label
behAll = behAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})
# Calculuate rewarded left option
leftCorrect = behAll['leftCanBePushed']*behAll.pushCorrect + (1-behAll['leftCanBePushed'])*(1-behAll.pushCorrect)
behAll['leftCorrect'] = leftCorrect
 
########################################### 1. Select option with higher amount 
leftCorrect_higher_amt = behAll['leftCorrect']*(behAll['winAmtLeft']>=50)*behAll['winAmtLeft']
rightCorrect_higher_amt  = (1-behAll['leftCorrect'])*(behAll['winAmtBlue']>=50)*behAll['winAmtBlue']
agent_higher_ammount = leftCorrect_higher_amt + rightCorrect_higher_amt
behAll['agent_higher_ammount'] = agent_higher_ammount 

############################################ 2. Select option with lower amount 
leftCorrect_lower_amt = behAll['leftCorrect']*(behAll['winAmtLeft']<50)*behAll['winAmtLeft']
rightCorrect_lower_amt  = (1-behAll['leftCorrect'])*(behAll['winAmtBlue']<50)*behAll['winAmtBlue']
agent_lower_ammount = leftCorrect_lower_amt + rightCorrect_lower_amt
behAll['agent_lower_ammount'] = agent_lower_ammount


###########################################  3. winStay/loseShift 
"""This agent takes winStay/loseShift for pushing/pulling  in action condition and yellowChosen/BlueChosen Color conditions"""
choiceWinLose = np.zeros(behAll.shape[0])
# 1 codes push/yellow and 0 codes pull/blue
for i in range(behAll.shape[0]-1):
    # set the typs pf rewarded option regarding to block
    if behAll['block'][i] =='Action':
        correct = 'pushCorrect'
    elif behAll['block'][i] =='Color':
        correct = 'yellowCorrect'
    # Check whether choice are 1 (push or yeloow) or 0 (pull or blue) and condition
    if (choiceWinLose[i]==1) and (choiceWinLose[i] == behAll[correct][i]):
        choiceWinLose[i + 1] = 1 # win stay (push or yellow)
    elif (choiceWinLose[i]==0) and (choiceWinLose[i] == behAll[correct][i]):
        choiceWinLose[i + 1] = 0 # win stay (pull or blue)
    elif (choiceWinLose[i]==1) and (choiceWinLose[i] != behAll[correct][i]):
        choiceWinLose[i + 1] = 0  # lose shift (push or yellow)
    elif (choiceWinLose[i]==0) and (choiceWinLose[i] != behAll[correct][i]):
        choiceWinLose[i + 1] = 1 # lose shift (pull or blue)

# Wom amount for agen 
behAll['agent_winstat_shiftlose'] = (behAll['block'] =='Action')*(choiceWinLose*behAll['pushCorrect']*behAll['winAmtPushable'] + (1-choiceWinLose)*(1-behAll['pushCorrect'])*behAll['winAmtPullable']) + (behAll['block'] =='Color')*(choiceWinLose*behAll['yellowCorrect']*behAll['winAmtYellow'] + (1-choiceWinLose)*(1-behAll['yellowCorrect'])*behAll['winAmtBlue']) 

########################################### 4. this agent choose randomly push and pull 
 # Select random choice, push coded 1 and pull coded 0
rand = np.random.binomial(1,.5, size = behAll.shape[0])
# pushed
behAll['agent_random'] = behAll['pushCorrect']*rand*behAll['winAmtPushable'] + (1-behAll['pushCorrect'])*(1-rand)*behAll['winAmtPullable']


########################################### 5. Agent chooses options with the higher probability of rewarding
# Rewarded choice for agen
behAll['choiceAgent_random'] = np.nan

# List of subjects
subList = behAll['sub_ID'].unique()

for subName in subList:
    for sess in [1,2]:
        for run in [1,2]:
            for cond in ['Color', 'Action']:
                
                # take data from specific session, run for a subject
                actData = behAll[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)]
                
                # number of phases
                phases = actData['phase'].unique()
                for phase in phases:
                    actDataPhase = actData[actData['phase']==phase]
                    if cond=='Action':
                        propPhase = actDataPhase['pushCorrect'].mean()
                    if cond=='Color':
                        propPhase = actDataPhase['yellowCorrect'].mean()
                    # Chossing option with higher probability reward
                    behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']==phase), 'choiceAgent_random'] = round(propPhase)

# Wom amount for agen
behAll['agent_hgiher_probability'] = (behAll['block'] =='Action')*(behAll['choiceAgent_random'] *behAll['pushCorrect']*behAll['winAmtPushable'] + (1-behAll['choiceAgent_random'] )*(1-behAll['pushCorrect'])*behAll['winAmtPullable']) + (behAll['block'] =='Color')*(behAll['choiceAgent_random'] *behAll['yellowCorrect']*behAll['winAmtYellow'] + (1-behAll['choiceAgent_random'] )*(1-behAll['yellowCorrect'])*behAll['winAmtBlue']) 

 
########################################### 5. Oracle Agent chooses options with higher expected values
# Rewarded choice for agen
behAll['agent_oracle'] = np.nan

# List of subjects
subList = behAll['sub_ID'].unique()

for subName in subList:
    for sess in [1,2]:
        for run in [1,2]:
            for cond in ['Color', 'Action']:
                # take data from specific session, run for a subject
                actData = behAll[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)]
                # number of phases
                phases = actData['phase'].unique()
                for phase in phases:
                    # select phase 
                    actDataPhase = actData[actData['phase']==phase]
                    if cond=='Action':
                        # To select push and pull in action value learning
                        propPhase = actDataPhase['pushCorrect'].mean()
                        # is expected value of amount pushable selected
                        EV_bool = propPhase*actDataPhase['winAmtPushable'].to_numpy()> (1-propPhase)*actDataPhase['winAmtPullable'].to_numpy()
                        # select maximum expected values
                        agent_oracle  = EV_bool*actDataPhase['pushCorrect']*actDataPhase['winAmtPushable'] + (1-EV_bool)*(1-actDataPhase['pushCorrect'])*actDataPhase['winAmtPullable']   
                        behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']==phase), 'agent_oracle'] = agent_oracle
                    elif cond=='Color':
                        # To select yellow and blue in color value learning
                        propPhase = actDataPhase['yellowCorrect'].mean()
                        EV_bool = propPhase*actDataPhase['winAmtYellow'].to_numpy()> (1-propPhase)*actDataPhase['winAmtBlue'].to_numpy()
                        agent_oracle  = EV_bool*actDataPhase['yellowCorrect']*actDataPhase['winAmtYellow'] + (1-EV_bool)*(1-actDataPhase['yellowCorrect'])*actDataPhase['winAmtBlue']   
                        behAll.loc[(behAll['session']==sess) & (behAll['run']==run) & (behAll['block']==cond) & (behAll['sub_ID']==subName)& (behAll['phase']==phase), 'agent_oracle'] = agent_oracle


############################################################# figure
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(21*mm, 12*mm), tight_layout=True)
#axs = axs.flatten()

custom_palette = {
    'HC': 'blue',
    'PD-ON': 'red',
    'PD-OFF': '#FF7F7F'  # Light red in hex
}

# average across subject in observation 
behAll_obs = behAll.groupby(['block', 'group', 'sub_ID'], as_index=False)['wonAmount'].sum()
behAll_obs.loc[behAll_obs['group']=='HC','wonAmount'] = behAll_obs[behAll_obs['group']=='HC']['wonAmount']/2
behAll_obs_act_HC = behAll_obs[(behAll_obs['block']=='Action') & (behAll_obs['group']=='HC')]
behAll_obs_act_PD_OFF = behAll_obs[(behAll_obs['block']=='Action') & (behAll_obs['group']=='PD-OFF')]
behAll_obs_act_PD_ON = behAll_obs[(behAll_obs['block']=='Action') & (behAll_obs['group']=='PD-ON')]
behAll_obs_clr_HC = behAll_obs[(behAll_obs['block']=='Color') & (behAll_obs['group']=='HC')]
behAll_obs_clr_PD_OFF = behAll_obs[(behAll_obs['block']=='Color') & (behAll_obs['group']=='PD-OFF')]
behAll_obs_clr_PD_ON = behAll_obs[(behAll_obs['block']=='Color') & (behAll_obs['group']=='PD-ON')]


# average across subject in agents 
behAll_agent = behAll.groupby(['block', 'sub_ID'], as_index=False)[['agent_higher_ammount', 'agent_lower_ammount', 'agent_winstat_shiftlose', 'agent_random', 'agent_hgiher_probability', 'agent_oracle']].sum()
behAll_agent_action = behAll_agent[behAll_agent['block']=='Action']
behAll_agent_color = behAll_agent[behAll_agent['block']=='Color']

# Define positions for each group and category
agent_positions = np.arange(6) * 2  # Base positions for each agent
offset = 0.3  # Offset for each category and dataset
positions_agen_action = agent_positions - offset  # action 
positions_agen_color = agent_positions + offset   # color


# action value learning for agent
means_agent_action= behAll_agent_action[['agent_lower_ammount', 'agent_winstat_shiftlose', 'agent_higher_ammount','agent_oracle', 'agent_hgiher_probability', 'agent_random']].mean(axis=0)/2
se_agent_action = behAll_agent_action[['agent_lower_ammount', 'agent_winstat_shiftlose', 'agent_higher_ammount','agent_oracle', 'agent_hgiher_probability', 'agent_random']].std(axis=0)  # Standard error of the mean
bars = axs.bar(positions_agen_action, means_agent_action, yerr=se_agent_action, 
               capsize=5, width=.5, color='red', edgecolor='black')

# Color value learning for agent
means_agent_color= behAll_agent_color[['agent_lower_ammount', 'agent_winstat_shiftlose', 'agent_higher_ammount','agent_oracle', 'agent_hgiher_probability', 'agent_random']].mean(axis=0)/2
se_agent_color = behAll_agent_color[['agent_lower_ammount', 'agent_winstat_shiftlose', 'agent_higher_ammount','agent_oracle', 'agent_hgiher_probability', 'agent_random']].std(axis=0)  # Standard error of the mean
bars = axs.bar(positions_agen_color, means_agent_color, yerr=se_agent_color, 
               capsize=5, width=.5, color='skyblue', edgecolor='black')

# position of real data
obs_positions = np.array([15,17,19])  # Base positions for each group
offset = 0.3  # Offset for each category and dataset
positions_obs_action = obs_positions - offset  # action 
positions_obs_color = obs_positions + offset   # color


# action value learning for real data
means_observation_action = [behAll_obs_act_HC['wonAmount'].mean(axis=0), 
                           behAll_obs_act_PD_OFF['wonAmount'].mean(axis=0),
                           behAll_obs_act_PD_ON['wonAmount'].mean(axis=0)]
se_observation_action = [behAll_obs_act_HC['wonAmount'].std(axis=0), 
                         behAll_obs_act_PD_OFF['wonAmount'].std(axis=0),
                         behAll_obs_act_PD_ON['wonAmount'].std(axis=0)]
bars = axs.bar(positions_obs_action, means_observation_action, yerr=se_observation_action, 
               capsize=5, width=.5, color='red', edgecolor='black', label='Action')


# Color value learning for real data
means_observation_color = [behAll_obs_clr_HC['wonAmount'].mean(axis=0), 
                           behAll_obs_clr_PD_OFF['wonAmount'].mean(axis=0),
                           behAll_obs_clr_PD_ON['wonAmount'].mean(axis=0)]
se_observation_color = [behAll_obs_clr_HC['wonAmount'].std(axis=0), 
                        behAll_obs_clr_PD_OFF['wonAmount'].std(axis=0),
                        behAll_obs_clr_PD_ON['wonAmount'].std(axis=0)]
bars = axs.bar(positions_obs_color, means_observation_color, yerr=se_observation_color, 
               capsize=5, width=.5, color='skyblue', edgecolor='black', label='Color')

# set the position and labels
axs.set_xticks(np.r_[agent_positions, obs_positions], 
               ['Lower Amt','Win-Stay/Shift-Lose','Higher Amt','Oracle','Probability','Random', 'HC','PD-OFF','PD-ON'],
               rotation=70)  
axs.set_ylim(0, 4500)
axs.set_ylabel('Total amount', fontsize=12)

axs.legend(title='Condition', loc='upper left')

# save plot
plt.savefig('../../Figures/simulation_vs_observation_plot.png', dpi=300)
  