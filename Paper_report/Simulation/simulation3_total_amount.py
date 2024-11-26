""" Simulation study without RL model.
Aget randomly selects Left/Right, Push/Pull, or Yellow/Pull in different simulations"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
rawBehAll['group'] = rawBehAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])

# Rename an column label
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})
# Calculuate rewarded left option
leftCorrect = rawBehAll['leftCanBePushed']*rawBehAll.pushCorrect + (1-rawBehAll['leftCanBePushed'])*(1-rawBehAll.pushCorrect)
rawBehAll['leftCorrect'] = leftCorrect
 
"""This agent choose randomly left or right"""
 # Select random choice 
rand = np.random.binomial(1,.5, size = rawBehAll.shape[0])
agentLeftRight = rand*rawBehAll['leftCorrect']*rawBehAll['winAmtLeft'] + (1-rand)*(1-rawBehAll['leftCorrect'])*rawBehAll['winAmtRight']
rawBehAll['agentLeftRight'] = agentLeftRight

"""This agent choose randomly push or pull"""
# Select random choice 
rand = np.random.binomial(1,.5, size = rawBehAll.shape[0])
agentPushPull = rand*rawBehAll['pushCorrect']*rawBehAll['winAmtPushable'] + (1-rand)*(1-rawBehAll['pushCorrect'])*rawBehAll['winAmtPullable']
rawBehAll['agentPushPull'] = agentPushPull

"""This agent choose randomly yellow or blue"""
# Select random choice 
rand = np.random.binomial(1,.5, size = rawBehAll.shape[0])
agentYellBlue = rand*rawBehAll['yellowCorrect']*rawBehAll['winAmtYellow'] + (1-rand)*(rawBehAll['yellowCorrect'])*rawBehAll['winAmtBlue']
rawBehAll['agentYellBlue'] = agentYellBlue
 

# figure
mm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(20*mm, 16*mm), tight_layout=True)
rows = 2
columns = 2

# Plot of left choice
fig.add_subplot(rows, columns, 1)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentLeftRight'].sum()
rawBehAll_wonAmount_agent.loc[rawBehAll_wonAmount_agent['group']=='HC','agentLeftRight'] = rawBehAll_wonAmount_agent[rawBehAll_wonAmount_agent['group']=='HC']['agentLeftRight']/2
sn = sns.barplot(data = rawBehAll_wonAmount_agent, x='group', y='agentLeftRight', hue='block', width=.5, errorbar="se", palette='Reds')
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act Obs','Clr Obs', 'Act Agent','Clr Agent']):
    t.set_text(l)
plt.title('Choosing left or right at random')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12)

# Plot of pushing
fig.add_subplot(rows, columns, 2)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_push_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentPushPull'].sum()
rawBehAll_push_agent.loc[rawBehAll_push_agent['group']=='HC','agentPushPull'] = rawBehAll_push_agent[rawBehAll_push_agent['group']=='HC']['agentPushPull']/2
sn = sns.barplot(data = rawBehAll_push_agent, x='group', y='agentPushPull', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Choosing push or pull at random')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12) 

# Plot of chosen yellow
fig.add_subplot(rows, columns, 3)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_yell_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentYellBlue'].sum()
rawBehAll_yell_agent.loc[rawBehAll_yell_agent['group']=='HC','agentYellBlue'] = rawBehAll_yell_agent[rawBehAll_yell_agent['group']=='HC']['agentYellBlue']/2
sn = sns.barplot(data = rawBehAll_yell_agent, x='group', y='agentYellBlue', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Choosing yellow or blue at random')
plt.ylabel('', fontsize=12)
plt.xlabel('', fontsize=12)

# common x and y label
fig.supxlabel('Group label', fontsize='12')
fig.supylabel('Total amount', fontsize='12')
#fig.suptitle('', fontsize='12')

# save figure 
plt.savefig('../Figures/simulation3_total_ammount.png', dpi=300)
plt.show()