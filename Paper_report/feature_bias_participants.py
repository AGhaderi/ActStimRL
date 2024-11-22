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
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['Condition'] = behAll['block'].replace(['Acyt', 'Stim'], ['Act', 'Clr'])

"""The same as the figure above but differnt action and color conditions.
"""

"""Left or right tendency across participant"""
left_groups= behAll.groupby(['group', 'Condition', 'sub_ID'], as_index=False)['leftChosen'].mean()

"""higher and lower amount tendency across participant"""
amt_groups= behAll.groupby(['group', 'Condition',  'sub_ID'], as_index=False)['chosenHighWinAmt'].mean()

"""push and pull tendency across participant"""
pushed_groups= behAll.groupby(['group', 'Condition',  'sub_ID'], as_index=False)['pushed'].mean()

"""yellow and blue tendency across participant"""
yellow_groups= behAll.groupby(['group', 'Condition',  'sub_ID'], as_index=False)['yellowChosen'].mean()


# plot of probability chosen left during trials
cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(20*cm, 16*cm), tight_layout=True)
row = 2
column = 2

# Chosen left
fig.add_subplot(row, column, 1)
ax = sns.barplot(
    data = left_groups, x='group', y='leftChosen', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=True)
sns.stripplot(
    data=left_groups, x='group', y='leftChosen', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)
plt.title('')
plt.xlabel('')
plt.ylabel('Left response', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')  
plt.ylim(0, 1)


# probability of higher amunt chosen
fig.add_subplot(row, column, 2)
ax = sns.barplot(
    data = amt_groups, x='group', y='chosenHighWinAmt', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)

sns.stripplot(
    data=amt_groups, x='group', y='chosenHighWinAmt', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)

plt.title('')
plt.xlabel('')
plt.ylabel('Chosen higher amount', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')
#plt.text(x=0, y=1, s='***')
#plt.text(x=1, y=1, s='***')
#plt.text(x=2, y=1, s='***')
plt.ylim(0, 1)
 
# probability of pushed
fig.add_subplot(row, column, 3)
ax = sns.barplot(
    data = pushed_groups, x='group', y='pushed', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)

sns.stripplot(
    data=pushed_groups, x='group', y='pushed', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)

plt.title('')
plt.xlabel('')
plt.ylabel('Pushed', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, 1)


# probability of yellow choisen
fig.add_subplot(row, column, 4)
ax = sns.barplot(
    data = yellow_groups, x='group', y='yellowChosen', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)

sns.stripplot(
    data=yellow_groups, x='group', y='yellowChosen', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)
plt.title('')
plt.xlabel('')
plt.ylabel('Chosen yellow', fontsize='12')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, 1)

fig.supxlabel('Group label')
fig.suptitle('Probability of chosing each feature across group and condition', fontsize='12')

# save figure
plt.savefig('Figures/feature_bias_participants.png', dpi=300)


plt.close()
# statsitical test over the proportion of each features
#test_amount = stats.ttest_1samp(amt_groups[(amt_groups['group']==2)&(amt_groups['Condition']=='Stim')]['chosenWinAmt'], .5)