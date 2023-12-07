"""Plot of choice behavior and rewarded point pattern for ach trial across all participants"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

# Read original action first amd color first task design to get reward schedule
dataActFirst = loadmat('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/predefined-task-values/ExpStruct_ActFirst_winOnly.mat')  
dataClrFirst = loadmat('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/predefined-task-values/ExpStruct_StimFirst_winOnly.mat')  

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

# Take a type of value learning and reversal envirnoment
block = 'Act'
reverse =  21
behAllCond = behAll.loc[(behAll['block']==block)&(behAll['reverse']==reverse)]

if block=='Stim' and reverse==21: 
    # Switch choice responses to proportional choices for Color value learning
    chosenOption = np.zeros(behAllCond.shape[0]) 
    chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['yellowChosen']-1)
    chosenOption[behAllCond['stimActFirst']=='Act'] = behAllCond.loc[behAllCond['stimActFirst']=='Act']['yellowChosen']
    behAllCond['highRewardOption'] = chosenOption
    
elif block=='Act' and reverse==21: 
    # Switch choice responses to proportional choices for Action value learning
    chosenOption = np.zeros(behAllCond.shape[0]) 
    chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['pushed']-1)
    chosenOption[behAllCond['stimActFirst']=='Act'] = behAllCond.loc[behAllCond['stimActFirst']=='Act']['pushed']
    behAllCond['highRewardOption'] = chosenOption

elif block=='Stim' and reverse==14:     
    # Switch choice responses to proportional choices for Color value learning
    chosenOption = np.zeros(behAllCond.shape[0]) 
    chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['yellowChosen']-1)
    chosenOption[behAllCond['stimActFirst']=='Act'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Act']['yellowChosen']-1)
    behAllCond['highRewardOption'] = chosenOption
    
elif block=='Act' and reverse==14: 
    # Switch choice responses to proportional choices for Color value learning
    chosenOption = np.zeros(behAllCond.shape[0]) 
    chosenOption[behAllCond['stimActFirst']=='Stim'] = behAllCond.loc[behAllCond['stimActFirst']=='Stim']['pushed']
    chosenOption[behAllCond['stimActFirst']=='Act'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Act']['pushed']-1)
    behAllCond['highRewardOption'] = chosenOption

# Window size for moving average
window_size = 4

"""High reward option"""
# Average of chosen options Across participants for each trial
behAllCond_chosenOption = behAllCond.groupby(['group', 'trialNumber'], as_index=False)['highRewardOption'].mean()
 
# Assignment of x and y values from the mean across subject for each trial
y_1_chosenOption = behAllCond_chosenOption[behAllCond_chosenOption['group']==1]['highRewardOption']
windows_y_1_chosenOption = y_1_chosenOption.rolling(window=window_size, min_periods=1)
moving_averages_y_1_chosenOption = windows_y_1_chosenOption.mean()
  
y_2_chosenOption = behAllCond_chosenOption[behAllCond_chosenOption['group']==2]['highRewardOption']
windows_y_2_chosenOption = y_2_chosenOption.rolling(window=window_size, min_periods=1)
moving_averages_y_2_chosenOption = windows_y_2_chosenOption.mean()

y_3_chosenOption = behAllCond_chosenOption[behAllCond_chosenOption['group']==3]['highRewardOption']
windows_y_3_chosenOption = y_3_chosenOption.rolling(window=window_size, min_periods=1)
moving_averages_y_3_chosenOption = windows_y_3_chosenOption.mean()


"""Rewarded points"""
# Average of rewarded points Across participants for each trial
behAllCond_correctChoice = behAllCond.groupby(['group', 'trialNumber'], as_index=False)['correctChoice'].mean()
  
# Assignment of x and y values from the mean across subject for each trial
y_1_correctChoice = behAllCond_correctChoice[behAllCond_correctChoice['group']==1]['correctChoice']
windows_y_1_correctChoice = y_1_correctChoice.rolling(window=window_size, min_periods=1)
moving_averages_y_1_correctChoice = windows_y_1_correctChoice.mean()
  
y_2_correctChoice = behAllCond_correctChoice[behAllCond_correctChoice['group']==2]['correctChoice']
windows_y_2_correctChoice = y_2_correctChoice.rolling(window=window_size, min_periods=1)
moving_averages_y_2_correctChoice = windows_y_2_correctChoice.mean()

y_3_correctChoice = behAllCond_correctChoice[behAllCond_correctChoice['group']==3]['correctChoice']
windows_y_3_correctChoice = y_3_correctChoice.rolling(window=window_size, min_periods=1)
moving_averages_y_3_correctChoice = windows_y_3_correctChoice.mean()
 
# plot 
fig = plt.figure(figsize=(10,8), tight_layout=True)
row = 2
column = 2

"""Chosen option"""

# The original plot
fig.add_subplot(row, column, 1)
plt.plot(np.arange(1, 43), y_1_chosenOption)
plt.plot(np.arange(1, 43), y_2_chosenOption)
plt.plot(np.arange(1, 43), y_3_chosenOption)
plt.xlabel('Trials', fontsize='12')
plt.ylabel('Chosen high rewarding option', fontsize='12')
plt.legend(['Group 1', 'Group 2', 'Group 3'])
plt.axhline(y=.5, color='black' , linestyle='--')
# add title
if block=='Act':
    plt.title('Action value learning')
else:
    plt.title('Color value learning')
# add reversal vertical line
if reverse==21:
    plt.axvline(x = 21, color='c', linestyle='--', linewidth=1, alpha=.7)
else:
    plt.axvline(x = 14, color='c', linestyle='--', linewidth=1, alpha=.7)
    plt.axvline(x = 28, color='c', linestyle='--', linewidth=1, alpha=.7)
# add rewarding schedule points
if block=='Act' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist2_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')
elif block=='Stim' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist1_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_1'][0]-1) + .03, s=10, c='#9467bd', marker='o')     
elif block=='Act' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist1_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), dataClrFirst['triallist2_1'][0] + .03, s=10, c='#9467bd', marker='o')     
elif block=='Stim' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist2_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist2_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')     

# The mosing average
fig.add_subplot(row, column, 2)
plt.plot(np.arange(1, 43), moving_averages_y_1_chosenOption)
plt.plot(np.arange(1, 43), moving_averages_y_2_chosenOption)
plt.plot(np.arange(1, 43), moving_averages_y_3_chosenOption)
plt.xlabel('Trials', fontsize='12')
plt.ylabel('Chosen high rewarding option', fontsize='12')
plt.axhline(y=.5, color='black' , linestyle='--')
# Add titles
if block=='Act':
    plt.title('Moving average ' + str(window_size) + ' - Action value learning')
else:
    plt.title('Moving average ' + str(window_size) + ' - Color value learning')
# add reversla vertical line
if reverse==21:
    plt.axvline(x = 21, color='c', linestyle='--', linewidth=1, alpha=.7)
else:
    plt.axvline(x = 14, color='c', linestyle='--', linewidth=1, alpha=.7)
    plt.axvline(x = 28, color='c', linestyle='--', linewidth=1, alpha=.7)
# add rewarding schedule points
if block=='Act' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist2_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')
elif block=='Stim' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist1_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_1'][0]-1) + .03, s=10, c='#9467bd', marker='o')     
elif block=='Act' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist1_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), dataClrFirst['triallist2_1'][0] + .03, s=10, c='#9467bd', marker='o')     
elif block=='Stim' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist2_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist2_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')     

    
"""Rewarded choice"""

# Original plot
fig.add_subplot(row, column, 3)
plt.plot(np.arange(1, 43), y_1_correctChoice)
plt.plot(np.arange(1, 43), y_2_correctChoice)
plt.plot(np.arange(1, 43), y_3_correctChoice)
plt.xlabel('Trials', fontsize='12')
plt.ylabel('Rewarded choice', fontsize='12')
plt.axhline(y=.5, color='black' , linestyle='--')
# add title
if block=='Act':
    plt.title('Action value learning')
else:
    plt.title('Color value learning')
# add reversal vertical line
if reverse==21:
    plt.axvline(x = 21, color='c', linestyle='--', linewidth=1, alpha=.7)
else:
    plt.axvline(x = 14, color='c', linestyle='--', linewidth=1, alpha=.7)
    plt.axvline(x = 28, color='c', linestyle='--', linewidth=1, alpha=.7)
# add rewarding schedule points
if block=='Act' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist2_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')
elif block=='Stim' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist1_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_1'][0]-1) + .03, s=10, c='#9467bd', marker='o')     
elif block=='Act' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist1_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), dataClrFirst['triallist2_1'][0] + .03, s=10, c='#9467bd', marker='o')     
elif block=='Stim' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist2_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist2_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')     



# Moving average
fig.add_subplot(row, column, 4)
plt.plot(np.arange(1, 43), moving_averages_y_1_correctChoice)
plt.plot(np.arange(1, 43), moving_averages_y_2_correctChoice)
plt.plot(np.arange(1, 43), moving_averages_y_3_correctChoice)
plt.xlabel('Trials', fontsize='12')
plt.ylabel('Rewarded choice', fontsize='12')
plt.axhline(y=.5, color='black' , linestyle='--')
# Add titles
if block=='Act':
    plt.title('Moving average ' + str(window_size) + ' - Action value learning')
else:
    plt.title('Moving average ' + str(window_size) + ' - Color value learning')
# add reversla vertical line
if reverse==21:
    plt.axvline(x = 21, color='c', linestyle='--', linewidth=1, alpha=.7)
else:
    plt.axvline(x = 14, color='c', linestyle='--', linewidth=1, alpha=.7)
    plt.axvline(x = 28, color='c', linestyle='--', linewidth=1, alpha=.7)
# add rewarding schedule points
if block=='Act' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist2_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')
elif block=='Stim' and reverse==21:
    plt.scatter(np.arange(1, 43), dataActFirst['triallist1_2'][0], s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist1_1'][0]-1) + .03, s=10, c='#9467bd', marker='o')     
elif block=='Act' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist1_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), dataClrFirst['triallist2_1'][0] + .03, s=10, c='#9467bd', marker='o')     
elif block=='Stim' and reverse==14:
    plt.scatter(np.arange(1, 43), -(dataActFirst['triallist2_1'][0]-1), s=10, c='#2ca02c', marker='o')
    plt.scatter(np.arange(1, 43), -(dataClrFirst['triallist2_2'][0]-1) + .03, s=10, c='#9467bd', marker='o')     

# Save
plt.savefig('../figures/plot_correct_and_choice_proportion_trials_' +block+ '_'+ str(reverse)+'.png', dpi=300)
plt.show()

# print out the proportion of chising with higher probability reward 

if reverse==21: 
    # select the first phase for 25 first trials
    behAllCond_phase1 = behAllCond[behAllCond['trialNumber']<=26].groupby(['group', 'sub_ID'], as_index=False)['highRewardOption'].mean()
    behAllCond_phase1_mean = behAllCond_phase1.groupby( ['group'])['highRewardOption'].mean()
    behAllCond_phase1_se = behAllCond_phase1.groupby( ['group'])['highRewardOption'].sem()
    # select the second phase fpr remaining trials
    behAllCond_phase2 = behAllCond[behAllCond['trialNumber']>26].groupby(['group', 'sub_ID'], as_index=False)['highRewardOption'].mean()
    behAllCond_phase2_mean = behAllCond_phase2.groupby( ['group'])['highRewardOption'].mean()
    behAllCond_phase2_se = behAllCond_phase2.groupby( ['group'])['highRewardOption'].sem()

    print( 'mean proportion - ' + block + ' condition and phase 1 of stable environemnt', behAllCond_phase1_mean)
    print( 'se proportion - ' + block + ' condition and phase 1 of stable environemnt', behAllCond_phase1_se)
    print( 'mean proportion - ' + block + ' condition and phase 2 of stable environemnt', 1-behAllCond_phase2_mean)
    print( 'se proportion - ' + block + ' condition and phase 2 of stable environemnt', behAllCond_phase2_se)
elif reverse==14:
    # select the first phase for 18 first trials
    behAllCond_phase1 = behAllCond[(behAllCond['trialNumber']>=4)&(behAllCond['trialNumber']<=19)].groupby(['group', 'sub_ID'], as_index=False).mean()
    behAllCond_phase1_mean = behAllCond_phase1.groupby( ['group'])['highRewardOption'].mean()
    behAllCond_phase1_se = behAllCond_phase1.groupby( ['group'])['highRewardOption'].sem()
    # select the second phase for between 18 to 33 trial
    behAllCond_phase2 = behAllCond[(behAllCond['trialNumber']>19)&(behAllCond['trialNumber']<=33)].groupby(['group', 'sub_ID'], as_index=False)['highRewardOption'].mean()
    behAllCond_phase2_mean = behAllCond_phase2.groupby( ['group'])['highRewardOption'].mean()
    behAllCond_phase2_se = behAllCond_phase2.groupby( ['group'])['highRewardOption'].sem()

    # select the third phase for the resting trials
    behAllCond_phase3 = behAllCond[behAllCond['trialNumber']>33].groupby(['group', 'sub_ID'], as_index=False)['highRewardOption'].mean()
    behAllCond_phase3_mean = behAllCond_phase3.groupby( ['group'])['highRewardOption'].mean()
    behAllCond_phase3_se = behAllCond_phase3.groupby( ['group'])['highRewardOption'].sem()

    print('mean proportion-  ' + block + ' condition and phase 1 of stable environemnt', behAllCond_phase1_mean)
    print('se proportion-  ' + block + ' condition and phase 1 of stable environemnt', behAllCond_phase1_se)


    print('mean proportion-  ' + block + ' condition and phase 2 of stable environemnt', 1-behAllCond_phase2_mean)
    print('se proportion-  ' + block + ' condition and phase 2 of stable environemnt', behAllCond_phase2_se)

    print('mean proportion-  ' + block + ' condition and phase 2 of stable environemnt', behAllCond_phase3_mean)
    print('se proportion-  ' + block + ' condition and phase 2 of stable environemnt', behAllCond_phase3_se)