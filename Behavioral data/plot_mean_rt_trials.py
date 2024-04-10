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

# Window size for moving average
window_size = 4


"""Rewarded points"""
# Average of rewarded points Across participants for each trial
behAllCond_correctChoice = behAllCond.groupby(['group', 'trialNumber'], as_index=False)['RT'].mean()
  
# Assignment of x and y values from the mean across subject for each trial
y_1_correctChoice = behAllCond_correctChoice[behAllCond_correctChoice['group']==1]['RT']
windows_y_1_correctChoice = y_1_correctChoice.rolling(window=window_size, min_periods=1)
moving_averages_y_1_correctChoice = windows_y_1_correctChoice.mean()
  
y_2_correctChoice = behAllCond_correctChoice[behAllCond_correctChoice['group']==2]['RT']
windows_y_2_correctChoice = y_2_correctChoice.rolling(window=window_size, min_periods=1)
moving_averages_y_2_correctChoice = windows_y_2_correctChoice.mean()

y_3_correctChoice = behAllCond_correctChoice[behAllCond_correctChoice['group']==3]['RT']
windows_y_3_correctChoice = y_3_correctChoice.rolling(window=window_size, min_periods=1)
moving_averages_y_3_correctChoice = windows_y_3_correctChoice.mean()
 
# plot 
fig = plt.figure(figsize=(10,8), tight_layout=True)
row = 1
column = 2

"""Rewarded choice"""

# Original plot
fig.add_subplot(row, column, 1)
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
fig.add_subplot(row, column, 2)
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
plt.savefig('../figures/plot_mean_rt_trials_' +block+ '_'+ str(reverse)+'.png', dpi=300)
plt.show()
