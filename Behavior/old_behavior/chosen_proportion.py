"""Intrinsic tendency of reward probability toward left/ right, push/pull, yellow/blue across participant for each condition."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')

print(rawBehAll.columns)

# Find chosen amount for each trial
chosenAmount = rawBehAll['leftChosen']*rawBehAll['winAmtLeft'] + (1-rawBehAll['leftChosen'])*rawBehAll['winAmtRight'] 
# Calculate the probability of high amount is chosed or lower amount
rawBehAll['chosenHighWinAmt'] = chosenAmount>=50
# Calculuate reward left option
leftCorrect = rawBehAll['leftCanBePushed                ']*rawBehAll.pushCorrect + (1-rawBehAll['leftCanBePushed                '])*(1-rawBehAll.pushCorrect)
rawBehAll['leftCorrect'] = leftCorrect

# the proportion of rewarded left nad push option in Action value learning
leftChosen_act = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Act')].groupby(['group', 'sub_ID'], as_index=False)['leftChosen'].mean()
pushed_act = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Act')].groupby(['group', 'sub_ID'], as_index=False)['pushed'].mean()
# the proportion of rewarded left and yellow option in Color value learning
leftChosen_stim = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Stim')].groupby(['group', 'sub_ID'], as_index=False)['leftChosen'].mean()
yellowChosen_stim = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Stim')].groupby(['group', 'sub_ID'], as_index=False)['yellowChosen'].mean()

# plot of probability reward in the task design
fig = plt.figure(figsize=(10,7), tight_layout=True)
nrows = 2
nvols = 2
 
# rewrading left
fig.add_subplot(nrows, nvols, 1)
sns.barplot(data = leftChosen_act, x='group', y='leftChosen',  width=.5, errorbar="se")
plt.title('a) Probability of chosen left in Act condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(chosen left in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .60)

# rewarding push
fig.add_subplot(nrows, nvols, 2)
sns.barplot(data = pushed_act, x='group', y='pushed',  width=.5, errorbar="se")
plt.title('c) Probability of chosen push in Act condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(chosen push in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .60)
 
# rewrding pull 
fig.add_subplot(nrows, nvols, 3)
sns.barplot(data = leftChosen_stim, x='group', y='leftChosen',  width=.5, errorbar="se")
plt.title('b) Probability of chosen left in Clr condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(chosen left in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .60)
 
# rewrding pull 
fig.add_subplot(nrows, nvols, 4)
sns.barplot(data = yellowChosen_stim, x='group', y='yellowChosen',  width=.5, errorbar="se")
plt.title('d) Probability of chosen yellow in Clr condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(chosen yellow in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .60)
# save figure
plt.savefig('../figures/chosen_proportion.png', dpi=300)
plt.show()
