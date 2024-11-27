"""
Trial-wise plot for eacg group labels.
This figure shows whether participants has a bias to choise left, pused, yellow chosed or higher amount for each trials.
This results disclose that higher amount is just the only sourse of bias.
The question is that, How and why this bias happened"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Find maximum and minimum for each trial
chosenAmount = behAll['leftChosen']*behAll['winAmtLeft'] + (1-behAll['leftChosen'])*behAll['winAmtRight'] 
# Calculate the probability of high amount is chosed or lower amount
behAll['chosenHighWinAmt'] = chosenAmount>=50

"""Left or right tendency during trials for each group"""
left_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['leftChosen'].mean()
left_group1 = left_groups[left_groups['group']==1]['leftChosen']
left_group2 = left_groups[left_groups['group']==2]['leftChosen']
left_group3 = left_groups[left_groups['group']==3]['leftChosen']

"""higher and lower amount tendency during trials for each group"""
amt_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['chosenHighWinAmt'].mean()
amt_group1 = amt_groups[amt_groups['group']==1]['chosenHighWinAmt']
amt_group2 = amt_groups[amt_groups['group']==2]['chosenHighWinAmt']
amt_group3 = amt_groups[amt_groups['group']==3]['chosenHighWinAmt']


"""push and pull tendency during trials for each group"""
pushed_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['pushed'].mean()
pushed_group1 = pushed_groups[pushed_groups['group']==1]['pushed']
pushed_group2 = pushed_groups[pushed_groups['group']==2]['pushed']
pushed_group3 = pushed_groups[pushed_groups['group']==3]['pushed']


"""yellow and blue tendency during trials for each group"""
yellow_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['yellowChosen'].mean()
yellow_group1 = yellow_groups[yellow_groups['group']==1]['yellowChosen']
yellow_group2 = yellow_groups[yellow_groups['group']==2]['yellowChosen']
yellow_group3 = yellow_groups[yellow_groups['group']==3]['yellowChosen']


# plot of probability chosen left during trials
fig = plt.figure(figsize=(10,7), tight_layout=True)
row = 2
column = 2

# probability of left chosen
fig.add_subplot(row, column, 1)
plt.plot(np.arange(1, 43), left_group1)
plt.plot(np.arange(1, 43), left_group2)
plt.plot(np.arange(1, 43), left_group3)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('Probability of left response for each trial')
plt.xlabel('Trials', fontsize='12')
plt.ylabel('P(left response)', fontsize='12')
plt.legend(['Group 1', 'Group 2', 'Group 3'])
plt.ylim(.2, .87)


# probability of higher amunt chosen
fig.add_subplot(row, column, 2)
plt.plot(np.arange(1, 43), amt_group1)
plt.plot(np.arange(1, 43), amt_group2)
plt.plot(np.arange(1, 43), amt_group3)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('Probability of chosen higher amount for each trial')
plt.xlabel('Trials', fontsize='12')
plt.ylabel('P(chosen higher amount)', fontsize='12')
plt.legend(['Group 1', 'Group 2', 'Group 3'])
plt.ylim(.2, .87)


# probability of pushed
fig.add_subplot(row, column, 3)
plt.plot(np.arange(1, 43), pushed_group1)
plt.plot(np.arange(1, 43), pushed_group2)
plt.plot(np.arange(1, 43), pushed_group3)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('Probability of push for each trial')
plt.xlabel('Trials', fontsize='12')
plt.ylabel('P(pushed)', fontsize='12')
plt.legend(['Group 1', 'Group 2', 'Group 3'])
plt.ylim(.2, .87)

# probability of yellow choisen
fig.add_subplot(row, column, 4)
plt.plot(np.arange(1, 43), yellow_group1)
plt.plot(np.arange(1, 43), yellow_group2)
plt.plot(np.arange(1, 43), yellow_group3)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('Probability of chosen yellow for each trial')
plt.xlabel('Trials', fontsize='12')
plt.ylabel('P(chosen yellow)', fontsize='12')
plt.legend(['Group 1', 'Group 2', 'Group 3'])
plt.ylim(.2, .87)

# save
plt.savefig('../figures/feature_bias_trial.png', dpi=300)
plt.show()