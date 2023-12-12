"""
This figure shows whether participants has a bias to choise left, pused, yellow chosed or higher amount across participants.
This results disclose that higher amount is just the only sourse of bias.
The question is that, How and why this bias happened"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')
# rearrange trial number
behAll['trialNumber'].replace(
       [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
       [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], inplace=True)

# Find chosen amount for each trial
chosenAmount = behAll['leftChosen']*behAll['winAmtLeft'] + (1-behAll['leftChosen'])*behAll['winAmtRight'] 
# Calculate the probability of high amount is chosed or lower amount
behAll['chosenHighWinAmt'] = chosenAmount>=50

"""The same as the figure above but differnt action and color conditions.
"""

"""Left or right tendency across participant"""
left_groups= behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['leftChosen'].mean()

"""higher and lower amount tendency across participant"""
amt_groups= behAll.groupby(['group', 'block',  'sub_ID'], as_index=False)['chosenHighWinAmt'].mean()

"""push and pull tendency across participant"""
pushed_groups= behAll.groupby(['group', 'block',  'sub_ID'], as_index=False)['pushed'].mean()

"""yellow and blue tendency across participant"""
yellow_groups= behAll.groupby(['group', 'block',  'sub_ID'], as_index=False)['yellowChosen'].mean()


# plot of probability chosen left during trials
fig = plt.figure(figsize=(10,7), tight_layout=True)
row = 2
column = 2

# Chosen left
fig.add_subplot(row, column, 1)
sn = sns.barplot(data = left_groups, x='group', y='leftChosen', hue='block', width=.5, errorbar="se")
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act', 'Clr']):
    t.set_text(l)
plt.title('Probability of left reponse across part.')
plt.xlabel('Group label', fontsize='12')
plt.ylabel('P(left response)', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')  
plt.ylim(0, .75)

# probability of higher amunt chosen
fig.add_subplot(row, column, 2)
sn = sns.barplot(data = amt_groups, x='group', y='chosenHighWinAmt', hue='block',  width=.5, errorbar="se")
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act', 'Clr']):
    t.set_text(l)
plt.title('Probability of chosen higher amount across part.')
plt.xlabel('Group label', fontsize='12')
plt.ylabel('P(chosen higher amount)', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')
plt.text(x=0, y=.7, s='***')
plt.text(x=1, y=.7, s='***')
plt.text(x=2, y=.7, s='***')
plt.ylim(0, .75)
 
# probability of pushed
fig.add_subplot(row, column, 3)
sn = sns.barplot(data = pushed_groups, x='group', y='pushed', hue='block',  width=.5, errorbar="se")
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act', 'Clr']):
    t.set_text(l)
plt.title('Probability of push across part.')
plt.xlabel('Group label', fontsize='12')
plt.ylabel('P(pushed)', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .75)


# probability of yellow choisen
fig.add_subplot(row, column, 4)
sn = sns.barplot(data = yellow_groups, x='group', y='yellowChosen', hue='block',  width=.5, errorbar="se")
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act', 'Clr']):
    t.set_text(l)
plt.title('Probability of chosen yellow across part.')
plt.xlabel('Group label', fontsize='12')
plt.ylabel('P(chosen yellow)', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .75)

# save figure
plt.savefig('../figures/feature_bias_participants.png', dpi=300)


plt.show()
# statsitical test over the proportion of each features
#test_amount = stats.ttest_1samp(amt_groups[(amt_groups['group']==2)&(amt_groups['block']=='Stim')]['chosenWinAmt'], .5)