"""Simulation study with Value-Maximizing policy.
Consirering two alpha parameters for two stable and volatile environemnt
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# the orginal task desing for each participatn
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
# Rename an column label
rawBehAll = rawBehAll.rename(columns={'wonAmount                ':'wonAmount'})
#slecect one reversal point environemnt
rawBehAll = rawBehAll[rawBehAll['reverse']==14]

# This function generates choice and rewarded choice based in alpha parameters defined in each trial 
# Simulation chooses push and pull in Action value learning condition and yellow and blue in COlor value leanring condition
def simulateActClr(task_design, simName):
    # List of subjects
    subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
               'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
               'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
               'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
               'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
               'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
               'sub-090', 'sub-092', 'sub-108', 'sub-109']
    # Loop over participatns and session, each simulation RL starts from the sratch for each participants and session
    for subName in subList:
        for ses in [1, 2]:
            task_sub_ses = task_design[(task_design['sub_ID']==subName) & (task_design['session']==ses)]
             # Predefined Number of trials
            n_trials = task_sub_ses.shape[0]
            # Predefined conditions for each trial
            block = task_sub_ses.block.to_numpy()
            # Predefined Winning amout of reward for Action and Color options
            winAmtPushable = task_sub_ses.winAmtPushable.to_numpy()
            winAmtPullable = task_sub_ses.winAmtPullable.to_numpy()
            winAmtYellow = task_sub_ses.winAmtYellow.to_numpy()
            winAmtBlue = task_sub_ses.winAmtBlue.to_numpy()  
            # Predefined Correct responces for Action and color options
            pushCorrect = task_sub_ses.pushCorrect.to_numpy()
            yellowCorrect = task_sub_ses.yellowCorrect.to_numpy()
            # Predefined Ground truth Parameters
            alpha = task_sub_ses['alpha'+ simName].to_numpy()
            # Output of simulation for correct choice and Action and Color chosen
            correctChoice = np.zeros(n_trials).astype(int)
            choice = np.zeros(n_trials).astype(int)
            wonAmount = np.zeros(n_trials).astype(int)
            # Initial reward probability
            
            probAct = .5
            probClr = .5
            # Loop over trials
            for i in range(n_trials):
                if block[i]=='Act':
                # Compute the Standard Expected Value of each seperated option 
                    expValue1 = probAct*winAmtPushable[i] 
                    expValue2 = probAct*winAmtPullable[i]
                    # Make a binary choice response with value-maximizing policy second step
                    y = int(expValue1>=expValue2)
                    choice[i] = y
                    # Get reward based on the simulated response, third step
                    correctChoice[i] = int(choice[i] == pushCorrect[i])
                    # Get won amount based on the simulated response
                    if choice[i]==1:
                        wonAmount[i] = correctChoice[i]*winAmtPushable[i]
                    elif choice[i] ==0:
                        wonAmount[i] = correctChoice[i]*winAmtPullable[i]
                    # Rl rule update over Action Learning Values for the next trial, fourth step
                    if choice[i] == 1:
                        probPush = probPush + alpha[i]*(correctChoice[i] - probPush)
                        probPull = 1 - probPush           
                    elif choice[i] == 0:
                        probPull = probPull + alpha[i]*(correctChoice[i] - probPull)
                        probPush = 1 - probPull                      
                elif block[i]=='Stim':
                    # Compute the Standard Expected Value of each seperated option 
                    expValue1 = probYell*winAmtYellow[i]
                    expValue2 = probBlue*winAmtBlue[i]
                    # Make a binary choice response with value-maximizing policy
                    y = int(expValue1>=expValue2)
                    choice[i] = y
                    # Get reward based on the simulated response 
                    correctChoice[i] = int(choice[i] == yellowCorrect[i])
                    # Get won amount based on the simulated response
                    if y==1:
                        wonAmount[i] = correctChoice[i]*winAmtYellow[i]
                    elif y==0:
                        wonAmount[i] = correctChoice[i]*winAmtBlue[i]
                    # Rl rule update Color Action Learning values for the next trial
                    if y == 1:
                        probYell = probYell + alpha[i]*(correctChoice[i] - probYell)
                        probBlue = 1 - probYell
                    elif y == 0:
                        probBlue = probBlue + alpha[i]*(correctChoice[i] - probBlue)
                        probYell = 1 - probBlue  
            # output results
            task_design.loc[(task_design['sub_ID']==subName) & (task_design['session']==ses), 'correctChoice_'+simName] = correctChoice
            task_design.loc[(task_design['sub_ID']==subName) & (task_design['session']==ses),'choice_'+simName] = choice
            task_design.loc[(task_design['sub_ID']==subName) & (task_design['session']==ses), 'wonAmount_'+simName] = wonAmount
    return task_design 

# Set the value of alpha parameters for simulating data from RL model
for i in np.linspace(0, .5, 16):
    n = round(i, 2)
    # Put the alpha value into a new column
    rawBehAll['alpha'+str(n)] = n
    # Call simulation function
    rawBehAll = simulateActClr(task_design = rawBehAll, simName=str(n))
 
# save subfigure
fig = plt.figure(figsize=(15, 15), tight_layout = True)
nrows= 4
ncols=4
idx = 1
for i in np.linspace(0, .5, 16):
    n = round(i, 2)
    fig.add_subplot(nrows, ncols, idx)
    plt.title('Alpha '+ str(n), fontsize='12')
    plt.ylim(0, 1700)

    # toral amount od simulated data by RL
    rawBehAll_wonAmount_agent = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount_'+ str(n)].sum()
    rawBehAll_wonAmount_agent.loc[rawBehAll_wonAmount_agent['group']==2,'wonAmount_'+ str(n)] = rawBehAll_wonAmount_agent[rawBehAll_wonAmount_agent['group']==2]['wonAmount_'+ str(n)]/2
    sns.barplot(data = rawBehAll_wonAmount_agent, x='group', y='wonAmount_'+ str(n), hue='block', width=.5, errorbar="se", palette='Reds')
    # total amount of observed data
    rawBehAll_wonAmount = rawBehAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
    rawBehAll_wonAmount.loc[rawBehAll_wonAmount['group']==2,'wonAmount'] = rawBehAll_wonAmount[rawBehAll_wonAmount['group']==2]['wonAmount']/2
    sn = sns.barplot(data = rawBehAll_wonAmount, x='group', y='wonAmount', hue='block', width=.5, errorbar="se", alpha=.9)
    if n==0.0:
        new_title = 'Condition'
        sn.legend_.set_title(new_title)
        for t, l in zip(sn.legend_.texts,['RL simulation Act','RL simulation Clr', 'Observed Act','Observed Clr']):
            t.set_text(l)
    else:
        sn.legend_.remove()
    plt.ylabel('')
    plt.xlabel('')
    # Change the label of legend
    idx+=1
# common x and y label
fig.text(0.5, 0, 'Group label', ha='center', fontsize='12')
fig.text(0, 0.5, 'Total amount', va='center', rotation='vertical', fontsize='12')

# Save figure
plt.savefig('../figures/simulation_rl_alpha_won_ammount_volatile.png', dpi=300)
plt.show()