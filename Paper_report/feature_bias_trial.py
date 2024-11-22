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
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['Condition'] = behAll['block'].replace(['Acyt', 'Stim'], ['Act', 'Clr'])

"""Left or right tendency during trials for each group"""
left_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['leftChosen'].mean()
left_HC = left_groups[left_groups['group']=='HC']['leftChosen']
left_PD_OFF = left_groups[left_groups['group']=='PD-OFF']['leftChosen']
left_PD_ON = left_groups[left_groups['group']=='PD-ON']['leftChosen']

"""higher and lower amount tendency during trials for each group"""
amt_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['chosenHighWinAmt'].mean()
amt_HC = amt_groups[amt_groups['group']=='HC']['chosenHighWinAmt']
amt_PD_OFF = amt_groups[amt_groups['group']=='PD-OFF']['chosenHighWinAmt']
amt_PD_ON = amt_groups[amt_groups['group']=='PD-ON']['chosenHighWinAmt']


"""push and pull tendency during trials for each group"""
pushed_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['pushed'].mean()
pushed_HC = pushed_groups[pushed_groups['group']=='HC']['pushed']
pushed_PD_OFF = pushed_groups[pushed_groups['group']=='PD-OFF']['pushed']
pushed_PD_ON = pushed_groups[pushed_groups['group']=='PD-ON']['pushed']


"""yellow and blue tendency during trials for each group"""
yellow_groups= behAll.groupby(['group', 'trialNumber'], as_index=False)['yellowChosen'].mean()
yellow_HC = yellow_groups[yellow_groups['group']=='HC']['yellowChosen']
yellow_PD_OFF = yellow_groups[yellow_groups['group']=='PD-OFF']['yellowChosen']
yellow_PD_ON = yellow_groups[yellow_groups['group']=='PD-ON']['yellowChosen']


# plot of probability chosen left during trials
mm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(20*mm, 16*mm), tight_layout=True)
row = 2
column = 2

# probability of left chosen
fig.add_subplot(row, column, 1)
plt.plot(np.arange(1, 43), left_HC)
plt.plot(np.arange(1, 43), left_PD_OFF)
plt.plot(np.arange(1, 43), left_PD_ON)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('')
plt.xlabel('')
plt.ylabel('Left response', fontsize='12')
plt.legend(['HC', 'PD-OFF', 'PD-ON'])
plt.ylim(.2, .9)
plt.xticks([1,10,20, 30, 42])


# probability of higher amunt chosen
fig.add_subplot(row, column, 2)
plt.plot(np.arange(1, 43), amt_HC)
plt.plot(np.arange(1, 43), amt_PD_OFF)
plt.plot(np.arange(1, 43), amt_PD_ON)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('')
plt.xlabel('')
plt.ylabel('Chosen higher amount', fontsize='12')
plt.ylim(.2, .9)
plt.xticks([1,10,20, 30, 42])


# probability of pushed
fig.add_subplot(row, column, 3)
plt.plot(np.arange(1, 43), pushed_HC)
plt.plot(np.arange(1, 43), pushed_PD_OFF)
plt.plot(np.arange(1, 43), pushed_PD_ON)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('')
plt.xlabel('')
plt.ylabel('Pushed', fontsize='12')
plt.ylim(.2, .9)
plt.xticks([1,10,20, 30, 42])

# probability of yellow choisen
fig.add_subplot(row, column, 4)
plt.plot(np.arange(1, 43), yellow_HC)
plt.plot(np.arange(1, 43), yellow_PD_OFF)
plt.plot(np.arange(1, 43), yellow_PD_ON)
plt.axhline(.5, color='black' , linestyle='--')
plt.title('')
plt.xlabel('')
plt.ylabel('Chosen yellow', fontsize='12')
plt.ylim(.2, .9)
plt.xticks([1,10,20, 30, 42])

fig.supxlabel('Trials')
fig.suptitle('Probability of chosing each feature across group, condition and trials', fontsize='12')


# save
plt.savefig('Figures/feature_bias_trial.png', dpi=300)
plt.close()