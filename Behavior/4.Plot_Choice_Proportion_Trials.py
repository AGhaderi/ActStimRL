"""Plot of choice behavior and rewarded point pattern for ach trial across all participants"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
 
# read collected data across data
# Main directory of the subject
readMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/'
# read collected data across all participants
behAll = pd.read_csv(f'{readMainDirec}/NoNanBehAll.csv')
# rearrange trial number
behAll['trialNumber'].replace(
       [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
       [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], inplace=True)

 
# plot 
mm = 1/2.54  # centimeters in inches
mm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*mm, 16*mm))
axs = axs.flatten()
# index of subplots
idx = 0
"""Take a type of value learning and reversal envirnoment""" 
for block in ['Act', 'Stim']:
    for reverse in [14,21]:
        # take the relavant data
        behAllCond = behAll.loc[(behAll['block']==block)&(behAll['reverse']==reverse)]

        if block=='Stim' and reverse==21: 
            # Switch choice responses to proportional choices for Color value learning
            chosenOption = np.zeros(behAllCond.shape[0]) 
            chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['yellowChosen']-1)
            chosenOption[behAllCond['stimActFirst']=='Act'] = behAllCond.loc[behAllCond['stimActFirst']=='Act']['yellowChosen']
            behAllCond['highRewardOption'] = chosenOption
            axs[idx].set_title('Color- One reversal point') 
      
        elif block=='Act' and reverse==21: 
            # Switch choice responses to proportional choices for Action value learning
            chosenOption = np.zeros(behAllCond.shape[0]) 
            chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['pushed']-1)
            chosenOption[behAllCond['stimActFirst']=='Act'] = behAllCond.loc[behAllCond['stimActFirst']=='Act']['pushed']
            behAllCond['highRewardOption'] = chosenOption
            axs[idx].set_title('Action- One reversal point') 

        elif block=='Stim' and reverse==14:     
            # Switch choice responses to proportional choices for Color value learning
            chosenOption = np.zeros(behAllCond.shape[0]) 
            chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['yellowChosen']-1)
            chosenOption[behAllCond['stimActFirst']=='Act'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Act']['yellowChosen']-1)
            behAllCond['highRewardOption'] = chosenOption
            axs[idx].set_title('Color- Two reversal points') 

        elif block=='Act' and reverse==14: 
            # Switch choice responses to proportional choices for Color value learning
            chosenOption = np.zeros(behAllCond.shape[0]) 
            chosenOption[behAllCond['stimActFirst']=='Stim'] = behAllCond.loc[behAllCond['stimActFirst']=='Stim']['pushed']
            chosenOption[behAllCond['stimActFirst']=='Act'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Act']['pushed']-1)
            behAllCond['highRewardOption'] = chosenOption
            axs[idx].set_title('Action- Two reversal points') 

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


        """Chosen option"""

        # The mosing average
        axs[idx].plot(np.arange(1, 43), moving_averages_y_2_chosenOption, color='blue')
        axs[idx].plot(np.arange(1, 43), moving_averages_y_1_chosenOption, color='#FF7F7F')
        axs[idx].plot(np.arange(1, 43), moving_averages_y_3_chosenOption, color='red')
        axs[idx].set_xlabel('', fontsize='12')
        axs[idx].set_ylabel('', fontsize='12')
        axs[idx].axhline(y=.5, color='black' , linestyle='--')
        axs[idx].set_xticks([1,10,20, 30, 42])

        # add reversal indication
        if reverse==21:
            axs[idx].axvline(x = 21, color='c', linestyle='--', linewidth=1, alpha=.7)
            axs[idx].plot([0, 21], [.75,.75], color='green')
            axs[idx].plot([21, 42], [.25,.25], color='green')
            axs[idx].plot([21, 21], [.75,.25], color='green')

        else:
            axs[idx].axvline(x = 14, color='c', linestyle='--', linewidth=1, alpha=.7)
            axs[idx].axvline(x = 28, color='c', linestyle='--', linewidth=1, alpha=.7)
            axs[idx].plot([0, 14], [.75,.75], color='green')
            axs[idx].plot([14, 14], [.75,.25], color='green')
            axs[idx].plot([14, 28], [.25,.25], color='green')
            axs[idx].plot([28, 28], [.75,.25], color='green')
            axs[idx].plot([28, 42], [.75,.75], color='green')

        if idx ==0:
            axs[idx].legend(['HC', 'PD-OFF', 'PD-ON'], fontsize=8)
        
        
        axs[idx].set_ylim(0,1)
        axs[idx].set_xlim(1,42)

        idx +=1
fig.supxlabel('Trials')
fig.supylabel('Choice Proportion', fontsize='12')

fig.tight_layout()

# Save
plt.savefig('/home/amingk/Documents/7TPD/Figures/Plot_Choice_Proportion_Trials.png', dpi=300)
plt.show()
