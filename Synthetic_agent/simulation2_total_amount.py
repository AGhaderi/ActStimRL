""" Simulation study without RL model.
This aget constantly pushes or pulls with higher amout and lower amount in different simulations."""

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

fig = plt.figure(figsize=(15, 5), tight_layout = True)
rows = 1
columns = 2

# Plot of left choice
fig.add_subplot(rows, columns, 1)
rawBehAll_high_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agent_higher_ammount'].sum()
rawBehAll_high_agent.loc[rawBehAll_high_agent['group']==2,'agent_higher_ammount'] = rawBehAll_high_agent[rawBehAll_high_agent['group']==2]['agent_higher_ammount']/2
sns.barplot(data = rawBehAll_high_agent, x='group', y='agent_higher_ammount', hue='block', width=.5, errorbar="se", palette='Reds')

rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
sn = sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Agent 2 Act','Agent 2 Clr', 'Observed Act','Observed Clr']):
    t.set_text(l)
plt.title('Choosing high-amount options constantly')
plt.ylabel('Total amount', fontsize=12)
plt.xlabel('Group label', fontsize=12)
plt.ylim(0, 3300)

#Plot of right choice
fig.add_subplot(rows, columns, 2)
rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se")

rawBehAll_low_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['agent_lower_ammount'].sum()
rawBehAll_low_agent.loc[rawBehAll_low_agent['group']==2,'agent_lower_ammount'] = rawBehAll_low_agent[rawBehAll_low_agent['group']==2]['agent_lower_ammount']/2
sn = sns.barplot(data = rawBehAll_low_agent, x='group', y='agent_lower_ammount', hue='block', width=.5, errorbar="se", palette='Reds')
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Observed Act','Observed Clr', 'Agent 2 Act','Agent 2 Clr']):
    t.set_text(l)
plt.title('Choosing low-amount options constantly')
plt.ylabel('Total amount', fontsize=12)
plt.xlabel('Group label', fontsize=12)
plt.ylim(0, 3300)

# save plot
plt.savefig('../figures/simulation2_total_ammount.png', dpi=300)
plt.show()

# proportiona of  rewarded higher and lowe amount
rawBehAll['higherCorrect'] = rawBehAll['leftCorrect']*(rawBehAll['winAmtLeft']>=50) + (1-rawBehAll['leftCorrect'])*(rawBehAll['winAmtLeft']>=50)
rawBehAll['lowerCorrect'] = rawBehAll['leftCorrect']*(rawBehAll['winAmtLeft']<50) + (1-rawBehAll['leftCorrect'])*(rawBehAll['winAmtLeft']<50)
propFeatures = rawBehAll.groupby(['group'])['higherCorrect', 'lowerCorrect'].mean()
print(propFeatures)