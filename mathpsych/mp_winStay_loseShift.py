"""Strategy of win stay and lose shift"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')

# Create empty dataframe to save proportion win stay and lose shift for each record
df_win_loss = pd.DataFrame()

# List of subjects
subList = np.unique(behAll.sub_ID)
for sub in subList:
    for session in range(2):
        for run in range(2):
            for block in ['Act', 'Stim']:
                
                # Take color or action value learning, one  
                behAllCond = behAll.loc[(behAll['session']==session+1)&(behAll['run']==run+1)&(behAll['block']==block)&(behAll['sub_ID']==sub)]
                # get reverse
                reverse = np.unique(behAllCond.reverse)[0]
                # get group label
                group = np.unique(behAllCond.group)[0]
                
                # number of winning reward for action value learning
                n_win = np.sum(behAllCond['correctChoice'].to_numpy()==1)
                win_stay = np.zeros(n_win)
                if block=='Act':
                    j = 0
                    for i in range(len(behAllCond)-1):
                        if behAllCond['correctChoice'].to_numpy()[i]==1:
                            win_stay[j] = behAllCond['pushed'].to_numpy()[i]==behAllCond['pushed'].to_numpy()[i+1]
                            j+=1
                else:
                    j = 0
                    for i in range(len(behAllCond)-1):
                        if behAllCond['correctChoice'].to_numpy()[i]==1:
                            win_stay[j] = behAllCond['yellowChosen'].to_numpy()[i]==behAllCond['yellowChosen'].to_numpy()[i+1]
                            j+=1

                # number of punishment for action value learning
                n_loss = np.sum(behAllCond['correctChoice'].to_numpy()==0)
                lose_stay = np.zeros(n_loss)
                if block=='Act':
                    j = 0
                    for i in range(len(behAllCond)-1):
                        if behAllCond['correctChoice'].to_numpy()[i]==0:
                            lose_stay[j] = behAllCond['pushed'].to_numpy()[i]==behAllCond['pushed'].to_numpy()[i+1]
                            j+=1
                else:
                    j = 0
                    for i in range(len(behAllCond)-1):
                        if behAllCond['correctChoice'].to_numpy()[i]==0:
                            lose_stay[j] = behAllCond['yellowChosen'].to_numpy()[i]==behAllCond['yellowChosen'].to_numpy()[i+1]
                            j+=1
                
                # dictionary for each record
                dic = {'session':session+1, 'run':run+1, 'block':block, 'sub_ID':sub, 'reverse':reverse, 'group':group, 
                      'win_stay': win_stay.mean(),  'lose_shift':1-lose_stay.mean()}
                # Put the dictionary into dataframe
                df_win_loss = pd.concat([pd.DataFrame([dic]), df_win_loss])
                

# Plot figures
fig = plt.figure(figsize=(10,4), tight_layout=True)
row = 1
column = 2

# Win stay
fig.add_subplot(row, column, 1)
sn = sns.barplot(data = df_win_loss, x='group', y='win_stay', hue='block',  width=.5, errorbar="sd")
new_title = 'Condition'
sn.legend_.set_title(new_title)
for t, l in zip(sn.legend_.texts,['Act', 'Clr']):
    t.set_text(l)
plt.title('Last trial rewarded- win stay', fontsize='12')
plt.xlabel('Group', fontsize='12')
plt.ylabel('P(stay|reward)', fontsize='12')
plt.xticks([0, 1, 2], ['OFF', 'HC', 'ON'])
plt.ylim(0, 1)

# Loss stay
fig.add_subplot(row, column, 2)
sn = sns.barplot(data = df_win_loss, x='group', y='lose_shift', hue='block',  width=.5, errorbar="sd")
sn.legend_.remove()
plt.title('Last trial no rewarded- lose shift', fontsize='12')
plt.xlabel('Group', fontsize='12')
plt.ylabel('P(stay|non-reward)', fontsize='12')
plt.xticks([0, 1, 2], ['OFF', 'HC', 'ON'])
plt.ylim(0, 1)

 # Save
plt.savefig('figures/mathpsych_winStay_loseShift.png', dpi=300)
plt.show()
