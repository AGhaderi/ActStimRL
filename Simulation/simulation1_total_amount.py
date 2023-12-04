""" Simulation study without RL model.
This aget constantly selects Left, right, Push, pull, Yellow, blue in differtn simulations.
When participants choose left options then they will get all rewarded point for left choices."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
# Rename an column label
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount', 'leftCanBePushed                ':'leftCanBePushed'})
# Calculuate rewarded left option
leftCorrect = rawBehAll['leftCanBePushed']*rawBehAll.pushCorrect + (1-rawBehAll['leftCanBePushed'])*(1-rawBehAll.pushCorrect)
rawBehAll['leftCorrect'] = leftCorrect

"""This agent select left option constantly"""
agentLeft = rawBehAll['leftCorrect']*rawBehAll['winAmtLeft']
rawBehAll['agentLeft'] = agentLeft
"""This agent select right option constantly"""
agentRight = (1-rawBehAll['leftCorrect'])*rawBehAll['winAmtRight']
rawBehAll['agentRight'] = agentLeft
"""This agent pushes constantly"""
agentPush = rawBehAll['pushCorrect']*rawBehAll['winAmtPushable']
rawBehAll['agentPush'] = agentPush
"""This agent pushes constantly"""
agentPull = (1-rawBehAll['pushCorrect'])*rawBehAll['winAmtPullable']
rawBehAll['agentPull'] = agentPull
"""This agent pushes constantly"""
agentYell = rawBehAll['yellowCorrect']*rawBehAll['winAmtYellow']
rawBehAll['agentYell'] = agentYell
"""This agent pushes constantly"""
agentBlue = (1-rawBehAll['yellowCorrect'])*rawBehAll['winAmtBlue']
rawBehAll['agentBlue'] = agentBlue

# Figure
fig = plt.figure(figsize=(10, 10), tight_layout = True)
nrows = 3
ncols = 2

# Plot of left choice
fig.add_subplot(nrows, ncols, 1)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentLeft'].sum()
rawBehAll_wonAmount_agent.loc[rawBehAll_wonAmount_agent['group']==2,'agentLeft'] = rawBehAll_wonAmount_agent[rawBehAll_wonAmount_agent['group']==2]['agentLeft']/2
sn = sns.barplot(data = rawBehAll_wonAmount_agent, x='group', y='agentLeft', hue='block', width=.5, errorbar="se", palette='Reds')
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Observed Act','Observed Clr', 'Agent 1 Act', 'Agent 1 Clr']):
    t.set_text(l)
plt.title('Choosing left option constantly')
plt.ylabel('')
plt.xlabel('')

#Plot of right choice
fig.add_subplot(nrows, ncols, 2)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_left_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentRight'].sum()
rawBehAll_left_agent.loc[rawBehAll_left_agent['group']==2,'agentRight'] = rawBehAll_left_agent[rawBehAll_left_agent['group']==2]['agentRight']/2
sn = sns.barplot(data = rawBehAll_left_agent, x='group', y='agentRight', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Choosing Right option constantly')
plt.ylabel('')
plt.xlabel('')

# Plot of pushing
fig.add_subplot(nrows, ncols, 3)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_push_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentPush'].sum()
rawBehAll_push_agent.loc[rawBehAll_push_agent['group']==2,'agentPush'] = rawBehAll_push_agent[rawBehAll_push_agent['group']==2]['agentPush']/2
sn = sns.barplot(data = rawBehAll_push_agent, x='group', y='agentPush', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Pushing constantly')
plt.ylabel('')
plt.xlabel('')

# Plot of pushing
fig.add_subplot(nrows, ncols, 4)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_pull_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentPull'].sum()
rawBehAll_pull_agent.loc[rawBehAll_pull_agent['group']==2,'agentPull'] = rawBehAll_pull_agent[rawBehAll_pull_agent['group']==2]['agentPull']/2
sn = sns.barplot(data = rawBehAll_pull_agent, x='group', y='agentPull', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Pulling constantly')
plt.ylabel('')
plt.xlabel('')

# Plot of chosen yellow
fig.add_subplot(nrows, ncols, 5)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_yell_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentYell'].sum()
rawBehAll_yell_agent.loc[rawBehAll_yell_agent['group']==2,'agentYell'] = rawBehAll_yell_agent[rawBehAll_yell_agent['group']==2]['agentYell']/2
sn = sns.barplot(data = rawBehAll_yell_agent, x='group', y='agentYell', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Choosing yellow constantly')
plt.ylabel('')
plt.xlabel('')

# Plot of pushing
fig.add_subplot(nrows, ncols, 6)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_blue_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentBlue'].sum()
rawBehAll_blue_agent.loc[rawBehAll_blue_agent['group']==2,'agentBlue'] = rawBehAll_blue_agent[rawBehAll_blue_agent['group']==2]['agentBlue']/2
sn = sns.barplot(data = rawBehAll_blue_agent, x='group', y='agentBlue', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Choosing blue constantly')
plt.ylabel('')
plt.xlabel('')

# common x and y label
fig.text(0.5, 0, 'Group label', ha='center', fontsize='12')
fig.text(0, 0.5, 'Total amount', va='center', rotation='vertical', fontsize='12')

plt.savefig('../figures/simulation1_total_ammount.png', dpi=300)
plt.show()