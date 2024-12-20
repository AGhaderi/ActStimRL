""" Simulation study without RL model.
This aget constantly selects Left, right, Push, pull, Yellow, blue in differtn simulations.
When participants choose left options then they will get all rewarded point for left choices."""

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
 
# plot  
mm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(21*mm, 12*mm), tight_layout=True)
nrows = 2
ncols = 3

# Plot of left choice
fig.add_subplot(nrows, ncols, 1)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentLeft'].sum()
rawBehAll_wonAmount_agent.loc[rawBehAll_wonAmount_agent['group']=='HC','agentLeft'] = rawBehAll_wonAmount_agent[rawBehAll_wonAmount_agent['group']=='HC']['agentLeft']/2
sn = sns.barplot(data = rawBehAll_wonAmount_agent, x='group', y='agentLeft', hue='block', width=.5, errorbar="se", palette='Reds')
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act Obs','Clr Obs', 'Act Agent', 'Clr Agent']):
    t.set_text(l)
plt.setp(sn.get_legend().get_texts(), fontsize='8')  
plt.setp(sn.get_legend().get_title(), fontsize='8')  

plt.title('Left', fontsize= '10')
plt.ylabel('')
plt.xlabel('')

#Plot of right choice
fig.add_subplot(nrows, ncols, 2)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_left_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentRight'].sum()
rawBehAll_left_agent.loc[rawBehAll_left_agent['group']=='HC','agentRight'] = rawBehAll_left_agent[rawBehAll_left_agent['group']=='HC']['agentRight']/2
sn = sns.barplot(data = rawBehAll_left_agent, x='group', y='agentRight', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Righ', fontsize= '10')
plt.ylabel('')
plt.xlabel('')

# Plot of pushing
fig.add_subplot(nrows, ncols, 3)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_push_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentPush'].sum()
rawBehAll_push_agent.loc[rawBehAll_push_agent['group']=='HC','agentPush'] = rawBehAll_push_agent[rawBehAll_push_agent['group']=='HC']['agentPush']/2
sn = sns.barplot(data = rawBehAll_push_agent, x='group', y='agentPush', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Pushed', fontsize= '10')
plt.ylabel('')
plt.xlabel('')

# Plot of pushing
fig.add_subplot(nrows, ncols, 4)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_pull_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentPull'].sum()
rawBehAll_pull_agent.loc[rawBehAll_pull_agent['group']=='HC','agentPull'] = rawBehAll_pull_agent[rawBehAll_pull_agent['group']=='HC']['agentPull']/2
sn = sns.barplot(data = rawBehAll_pull_agent, x='group', y='agentPull', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Pulled', fontsize= '10')
plt.ylabel('')
plt.xlabel('')

# Plot of chosen yellow
fig.add_subplot(nrows, ncols, 5)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_yell_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentYell'].sum()
rawBehAll_yell_agent.loc[rawBehAll_yell_agent['group']=='HC','agentYell'] = rawBehAll_yell_agent[rawBehAll_yell_agent['group']=='HC']['agentYell']/2
sn = sns.barplot(data = rawBehAll_yell_agent, x='group', y='agentYell', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Chosen yellow', fontsize= '10')
plt.ylabel('')
plt.xlabel('')

# Plot of pushing
fig.add_subplot(nrows, ncols, 6)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']=='HC','wonAmount'] = rawBehAll_wonAmount[rawBehAll_push_agent['group']=='HC']['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_blue_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agentBlue'].sum()
rawBehAll_blue_agent.loc[rawBehAll_blue_agent['group']=='HC','agentBlue'] = rawBehAll_blue_agent[rawBehAll_blue_agent['group']=='HC']['agentBlue']/2
sn = sns.barplot(data = rawBehAll_blue_agent, x='group', y='agentBlue', hue='block', width=.5, errorbar="se", palette='Reds')
sn.legend_.remove()
plt.title('Chosen Blue')
plt.ylabel('')
plt.xlabel('')

# common x and y label
fig.supxlabel('Group label', fontsize='12')
fig.supylabel('Total amount', fontsize='12')
#fig.suptitle('', fontsize='12')


plt.savefig('../Figures/simulation1_total_ammount.png', dpi=300)
plt.show()
# proportiona of left, push and yellow across participants
propFeatures = rawBehAll.groupby(['group'])[['leftCorrect', 'pushCorrect', 'yellowCorrect']].mean()
print(propFeatures)