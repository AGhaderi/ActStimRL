"""
This figure shows whether participants has a bias to choise left, pused, yellow chosed or higher amount across participants.
This results disclose that higher amount is just the only sourse of bias.
The question is that, How and why this bias happened"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/BehData/AllBehData/behAll.csv')
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
behAll['Condition'] = behAll['block'].replace(['Act', 'Stim'], ['Action', 'Color'])

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



# plot of learning rate in relavant condition
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*mm, 16*mm))
axs = axs.flatten()
#fig.subplots_adjust(wspace=.3) 
# Define the color palette for the hue categories
custom_palette = {
    'HC': 'blue',
    'PD-ON': 'red',
    'PD-OFF': '#FF7F7F'  # Light red in hex
}

# Chosen left
ax = sns.violinplot( data = left_groups, x='Condition', y='leftChosen', hue='group', ax=axs[0], legend=True, palette=custom_palette)
axs[0].set_title('')
axs[0].set_xlabel('')
axs[0].set_ylabel('Left reponse', fontsize='12')
axs[0].axhline(.5, color='black' , linestyle='--')  
axs[0].set_ylim(0, 1.2)
axs[0].legend(fontsize=8, loc='upper left')


# probability of higher amunt chosen
ax = sns.violinplot( data = amt_groups, x='Condition', y='chosenHighWinAmt', hue='group', ax=axs[1], legend=False, palette=custom_palette)
axs[1].set_title('')
axs[1].set_xlabel('')
axs[1].set_ylabel('Higher amount', fontsize='12')
axs[1].axhline(.5, color='black' , linestyle='--')  
axs[1].set_ylim(0, 1.2)

 
# probability of pushed
ax = sns.violinplot( data = pushed_groups, x='Condition', y='pushed', hue='group', ax=axs[2], legend=False, palette=custom_palette)
axs[2].set_title('')
axs[2].set_xlabel('')
axs[2].set_ylabel('Push reponse', fontsize='12')
axs[2].axhline(.5, color='black' , linestyle='--')  
axs[2].set_ylim(0, 1.2)

  

# probability of yellow choisen
ax = sns.violinplot( data = yellow_groups, x='Condition', y='yellowChosen', hue='group', ax=axs[3], legend=False, palette=custom_palette)
axs[3].set_title('')
axs[3].set_xlabel('')
axs[3].set_ylabel('Yellow reponse', fontsize='12')
axs[3].axhline(.5, color='black' , linestyle='--')  
axs[3].set_ylim(0, 1.2)

fig.supxlabel('Condition')
fig.suptitle('Probability of chosing each feature across group and condition', fontsize='12')

# adjust plot
fig.tight_layout()

# save figure
plt.savefig('../../Figures/feature_bias_participants.png', dpi=300)
 