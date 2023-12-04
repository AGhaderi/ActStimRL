"""Intrinsic tendency of reward probability toward left/ right, push/pull, yellow/blue across participant for each condition."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
 
# Find chosen amount for each trial
chosenAmount = rawBehAll['leftChosen']*rawBehAll['winAmtLeft'] + (1-rawBehAll['leftChosen'])*rawBehAll['winAmtRight'] 
# Calculate the probability of high amount is chosed or lower amount
rawBehAll['chosenHighWinAmt'] = chosenAmount>=50
# Calculuate reward left option
leftCorrect = rawBehAll['leftCanBePushed                ']*rawBehAll.pushCorrect + (1-rawBehAll['leftCanBePushed                '])*(1-rawBehAll.pushCorrect)
rawBehAll['leftCorrect'] = leftCorrect

# the proportion of rewarded left nad push option in Action value learning
leftCorrect_act = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Act')].groupby(['group', 'sub_ID'], as_index=False)['leftCorrect'].mean()
pushCorrect_act = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Act')].groupby(['group', 'sub_ID'], as_index=False)['pushCorrect'].mean()
# the proportion of rewarded left and yellow option in Color value learning
leftCorrect_stim = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Stim')].groupby(['group', 'sub_ID'], as_index=False)['leftCorrect'].mean()
yellowCorrect_stim = rawBehAll[(rawBehAll['chosenHighWinAmt']==1)&(rawBehAll['block']=='Stim')].groupby(['group', 'sub_ID'], as_index=False)['yellowCorrect'].mean()

# plot of probability reward in the task design
fig = plt.figure(figsize=(10,7), tight_layout=True)
nrows = 2
nvols = 2
 
# rewrading left
fig.add_subplot(nrows, nvols, 1)
sns.barplot(data = leftCorrect_act, x='group', y='leftCorrect',  width=.5, errorbar="se")
plt.title('a) Probability of rewarded left in Act condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(rewarded left in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .55)

# rewarding push
fig.add_subplot(nrows, nvols, 2)
sns.barplot(data = pushCorrect_act, x='group', y='pushCorrect',  width=.5, errorbar="se")
plt.title('c) Probability of rewarded push in Act condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(rewarded push in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .55)
 
# rewrding pull 
fig.add_subplot(nrows, nvols, 3)
sns.barplot(data = leftCorrect_stim, x='group', y='leftCorrect',  width=.5, errorbar="se")
plt.title('b) Probability of rewarded left in Clr condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(rewarded left in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .55)
 
# rewrding pull 
fig.add_subplot(nrows, nvols, 4)
sns.barplot(data = yellowCorrect_stim, x='group', y='yellowCorrect',  width=.5, errorbar="se")
plt.text(x=0, y=.51, s='***')
plt.text(x=1, y=.51, s='***')
plt.text(x=2, y=.51, s='*')
plt.title('d) Probability of rewarded yellow in Clr condition')
plt.xlabel('Group label', fontsize='10')
plt.ylabel('P(rewarded yellow in higher amt)', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .55)
plt.ylim(0, .55)
# save figure
plt.savefig('../figures/rewarded_options_amount.png', dpi=300)
plt.show()
