#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
"""all arguments have 2*2 array dimension"""
import numpy as np
import pandas as pd
import os
from scipy import stats
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import plots
import json

def generating_hier_grand_truth(hier_weight_mu, hier_alphaAct_pos_mu, hier_alphaAct_neg_mu,
                                hier_alphaClr_pos_mu, hier_alphaClr_neg_mu, hier_sensitivity_mu,
                                hier_alpha_sd, hier_weight_sd,  hier_sensitivity_sd,
                                simNumber):
    """generate and put individual and heirarchical true parameters into task desgin for each participant"""
    try:
        # read collected data across data
        rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/AllBehData/rawBehAll.csv')
        # list of subjects
        subList = rawBehAll['sub_ID'].unique()
        # Set true parameters for each session and conditions realted to unkown parameters
        for subName in subList:
            # Get the partisipant's task design from the original behavioral dataset 'originalfMRIbehFiles'
            rawBehAll = rawBehAll.rename(columns={'leftCanBePushed                ': 'leftCanBePushed'})
            task_design = rawBehAll[rawBehAll['sub_ID']==subName]
            # choose some relevant columns
            task_design = task_design[['session', 'run', 'stimActFirst', 'block', 'stimActBlock', 'trialNumber', 'yellowOnLeftSide', 'leftCanBePushed', 'winAmtLeft', 'winAmtRight', 'winAmtYellow', 'winAmtBlue', 'winAmtPushable', 'winAmtPullable', 'yellowCorrect', 'pushCorrect', 'reverse', 'group', 'patient']]
            # generate new samples from hierarhcial parameters
            transfer_weight = stats.norm.cdf(np.random.normal(hier_weight_mu,hier_weight_sd))
            transfer_alphaAct_pos = stats.norm.cdf(np.random.normal(hier_alphaAct_pos_mu,hier_alpha_sd))
            transfer_alphaAct_neg = stats.norm.cdf(np.random.normal(hier_alphaAct_neg_mu,hier_alpha_sd))
            transfer_alphaClr_pos = stats.norm.cdf(np.random.normal(hier_alphaClr_pos_mu,hier_alpha_sd))
            transfer_alphaClr_neg = stats.norm.cdf(np.random.normal(hier_alphaClr_neg_mu,hier_alpha_sd))
            transfer_sensitivity = np.log(1 + np.exp(np.random.normal(hier_sensitivity_mu,hier_sensitivity_sd)))
            # Put true parameters into the task design, define new columns of grand truth parameters within predefined task design 
            task_design[['transfer_alphaAct_pos', 'transfer_alphaAct_neg', 'transfer_alphaClr_pos', 'transfer_alphaClr_neg', 'transfer_weight', 'transfer_sensitivity']] = ""  
            # Put generated true parameters within the predefined task design dataframe
            for patient_index, patient in enumerate(['HC', 'PD']):
                for condition, block in enumerate(['Act', 'Stim']):
                    task_design.loc[(task_design['patient'] == patient)& (task_design['block'] == block), 'transfer_alphaAct_pos'] = transfer_alphaAct_pos[patient_index]
                    task_design.loc[(task_design['patient'] == patient)& (task_design['block'] == block), 'transfer_alphaAct_neg'] = transfer_alphaAct_neg[patient_index]
                    task_design.loc[(task_design['patient'] == patient)& (task_design['block'] == block), 'transfer_alphaClr_pos'] = transfer_alphaClr_pos[patient_index]
                    task_design.loc[(task_design['patient'] == patient)& (task_design['block'] == block), 'transfer_alphaClr_neg'] = transfer_alphaClr_neg[patient_index]
                    task_design.loc[(task_design['patient'] == patient)& (task_design['block'] == block), 'transfer_weight'] = transfer_weight[patient_index, condition]
                    task_design.loc[(task_design['patient'] == patient)& (task_design['block'] == block), 'transfer_sensitivity'] = transfer_sensitivity[patient_index]
            # Directory of simulated data
            parent_dir = '/mnt/scratch/projects/7TPD/amin/simulation'
            # Check existing directory of subject name forlder and simulation number
            if not os.path.isdir(f'{parent_dir}/{str(simNumber)}'):
                os.makedirs(f'{parent_dir}/{str(simNumber)}') 
            if not os.path.isdir(f'{parent_dir}/{str(simNumber)}/{subName}'):
                os.makedirs(f'{parent_dir}/{str(simNumber)}/{subName}')
            # Save task design plus true parameters for each participant
            task_design.to_csv(f'{parent_dir}/{str(simNumber)}/{subName}/{subName}-task-design-true-param.csv', index=False)
       
        # datafram of hierarchical true parameters
        dictionary =  {'hier_alphaAct_pos_mu':hier_alphaAct_pos_mu,
                      'hier_alphaAct_neg_mu': hier_alphaAct_neg_mu,
                      'hier_alphaClr_pos_mu': hier_alphaClr_pos_mu,
                      'hier_alphaClr_neg_mu': hier_alphaClr_neg_mu,
                      'hier_weight_mu': hier_weight_mu,
                      'hier_sensitivity_mu': hier_sensitivity_mu,
                      'hier_alpha_sd': hier_alpha_sd,
                      'hier_weight_sd': hier_weight_sd,
                      'hier_sensitivity_sd': hier_sensitivity_sd}
        # Writing to sample.json
        with open(f'{parent_dir}/{str(simNumber)}/hier-true-param.json', 'w') as outfile:
            json.dump(dictionary, outfile)

        return print("All true parameters for each participant have been generated and saved successfully!")
    except Exception as e:
        return print("An exception accured within generating_hier_grand_truth function: " + str(e))

def simulate_hier_rl(simNumber):
    """Simulated data for each participatn based on predefined True Parameters"""
    # read opbervation to take out the participnats
    rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/AllBehData/rawBehAll.csv')
    # list of subjects
    subList = rawBehAll['sub_ID'].unique()
    # Directory of simulated data
    parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation'

    # Simulation for participant
    for subName in subList:
        # Read predefined task design with true parameters
        task_design = pd.read_csv(f'{parent_dir}/{str(simNumber)}/{subName}/{subName}-task-design-true-param.csv')
        # simulate data
        simulated_data = simulate_rl(task_design)
        simulated_data.to_csv(f'{parent_dir}/{str(simNumber)}/{subName}/{subName}-sim-task-design-true-param.csv', index=False)
    
    return print("All simulations have been done successfully!")
    
def simulate_rl(task_design):
    """Simulated data from the predefined true parameters in dataframe task_design_param"""
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Stim']: # condition
                # get some relevant part data
                task_design_param_split = task_design[(task_design['block']==condition)&(task_design['session']==session)&(task_design['reverse']==reverse)]  

                # Predefined conditions for each trial
                block = task_design_param_split.block.to_numpy()
                
                # Predefined Winning amout of reward for Action and Color options
                winAmtPushable = task_design_param_split['winAmtPushable'].to_numpy()
                winAmtPullable = task_design_param_split['winAmtPullable'].to_numpy()
                winAmtYellow = task_design_param_split['winAmtYellow'].to_numpy()
                winAmtBlue = task_design_param_split['winAmtBlue'].to_numpy()  
                
                # Predefined options on left and right side
                leftCanBePushed = task_design_param_split['leftCanBePushed'].to_numpy()
                yellowOnLeftSide = task_design_param_split['yellowOnLeftSide'].to_numpy()
                
                # Predefined Correct responces for Action and color options
                pushCorrect = task_design_param_split['pushCorrect'].to_numpy()
                yellowCorrect = task_design_param_split['yellowCorrect'].to_numpy()
                
                # Predefined Ground truth Parameters
                transfer_alphaAct_pos = task_design_param_split['transfer_alphaAct_pos'].to_numpy()
                transfer_alphaAct_neg = task_design_param_split['transfer_alphaAct_neg'].to_numpy()
                transfer_alphaClr_pos = task_design_param_split['transfer_alphaClr_pos'].to_numpy()
                transfer_alphaClr_neg = task_design_param_split['transfer_alphaClr_neg'].to_numpy()
                transfer_weight = task_design_param_split['transfer_weight'].to_numpy()
                transfer_sensitivity = task_design_param_split['transfer_sensitivity'].to_numpy()

                
                # Predefined Number of trials
                n_trials = task_design_param_split.shape[0]
 
                # Output of simulation for correct choice and Action and Color chosen
                correctChoice = np.zeros(n_trials).astype(int)
                pushed = np.zeros(n_trials).astype(int)
                yellowChosen = np.zeros(n_trials).astype(int)

                # Initial reward probability
                p_push = .5
                p_pull = .5
                p_yell = .5
                p_blue = .5
                    
                # Loop over trials
                for i in range(n_trials):
                    
                    # Compute the Standard Expected Value of each seperated option 
                    EV_push = p_push*winAmtPushable[i]
                    EV_pull = p_pull*winAmtPullable[i]
                    EV_yell = p_yell*winAmtYellow[i]
                    EV_blue = p_blue*winAmtBlue[i]

                    # Relative contribution of Action Value LeexpValuePusharning verus Color Value Learning by combining the expected values of option
                    EV_push_yell = transfer_weight[i]*EV_push + (1 - transfer_weight[i])*EV_yell;
                    EV_push_blue = transfer_weight[i]*EV_push + (1 - transfer_weight[i])*EV_blue;
                    EV_pull_yell = transfer_weight[i]*EV_pull + (1 - transfer_weight[i])*EV_yell;
                    EV_pull_blue = transfer_weight[i]*EV_pull + (1 - transfer_weight[i])*EV_blue;

                    # Calculating the soft-max function based on (pushed and yellow) vs (pulled and blue) 
                    if (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 1) or (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 0):
                        # Applying soft-max function 
                        nom = np.exp(transfer_sensitivity[i]*EV_push_yell)
                        denom = nom + np.exp(transfer_sensitivity[i]*EV_pull_blue)
                        theta = nom/denom
                        # Make a binary choice response by bernouli 
                        y = np.random.binomial(1, p=theta, size=1) 
                        # Calculating to which Action vs Color Response response
                        if y==1:
                            pushed[i] = 1
                            yellowChosen[i] = 1
                        else:
                            pushed[i] = 0
                            yellowChosen[i] = 0
                    # Calculating the soft-max function based on (pushed and blue) vs (pulled and yellow) 
                    elif (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 0) or (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 1):
                        # Applying soft-max function 
                        nom = np.exp(transfer_sensitivity[i]*EV_push_blue)
                        denom = nom + np.exp(transfer_sensitivity[i]*EV_pull_yell)
                        theta = nom/denom
                        # Make a binary choice response by bernouli 
                        y = np.random.binomial(1, p=theta, size=1)
                        # Make a choice based on the probability 
                        if y==1:
                            pushed[i] = 1
                            yellowChosen[i] = 0
                        else:
                            pushed[i] = 0
                            yellowChosen[i] = 1

                    if block[i] == 'Act':
                        # Get reward based on the simulated response
                        correctChoice[i] = int(pushed[i] == pushCorrect[i])
                        
                        # Rl rule update over Action Learning Values for the next trial
                        if pushed[i] == 1:
                            if correctChoice[i]:
                                p_push = p_push + transfer_alphaAct_pos[i]*(correctChoice[i] - p_push)
                            else:
                                p_push = p_push + transfer_alphaAct_neg[i]*(correctChoice[i] - p_push)
                        elif pushed[i] == 0:
                            if correctChoice[i]>0:
                                p_pull = p_pull + transfer_alphaAct_pos[i]*(correctChoice[i] - p_pull)
                            else:
                                p_pull = p_pull + transfer_alphaAct_neg[i]*(correctChoice[i] - p_pull)
                                
                    elif block[i] == 'Stim':
                        correctChoice[i] = int(yellowChosen[i] == yellowCorrect[i])

                            # Rl rule update Color Action Learning values for the next trial
                        if yellowChosen[i] == 1:
                            if correctChoice[i]>0:
                                p_yell = p_yell + transfer_alphaClr_pos[i]*(correctChoice[i] - p_yell)
                            else:
                                p_yell = p_yell + transfer_alphaClr_neg[i]*(correctChoice[i] - p_yell)
                        elif yellowChosen[i] == 0:
                            if correctChoice[i]>0:
                                p_blue = p_blue + transfer_alphaClr_pos[i]*(correctChoice[i] - p_blue)
                            else:
                                p_blue = p_blue + transfer_alphaClr_neg[i]*(correctChoice[i] - p_blue)
                 

                # output results
                task_design.loc[(task_design['block']==condition)&(task_design['session']==session)&(task_design['reverse']==reverse), 'pushed'] = pushed  
                task_design.loc[(task_design['block']==condition)&(task_design['session']==session)&(task_design['reverse']==reverse), 'yellowChosen'] = yellowChosen  
                task_design.loc[(task_design['block']==condition)&(task_design['session']==session)&(task_design['reverse']==reverse), 'correctChoice'] = correctChoice  
 
    return task_design

"""Generate True values of Hierarchical parameters"""
# Simulation number
simNumber = 3

# grand truth of mean paramaters for each parameter
weight_Act = np.array([.4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])
weight_Clr = np.array([-.4, -.5, -.6,  -.7, -.8, -.9, -1, -1.1, -1.2, -1.3, -1.4, -1.5])
learning_rate = np.array([-1.6, -1.3, -1, -.7, -.6, -.3, 0, .3, .6 , .9, 1.2, 1.5])
sensitivity = np.array([-5, -4.8, -4.6, -4.2, -4, -3.8, -3.6, -3.4, -3.2 , -3, -2.8, -2.6])

# grand truth of sd paramaters for each parameter
alpha_sd = np.array([.01, .04, .07, .1, .13, .16, .19, .22, .25, .23, .26, .27])
weight_sd = np.array([.01, .04, .07, .1, .13, .16, .19, .22, .25, .23, .26, .27])
sensitivity_sd = np.array([.01, .04, .07, .1, .13, .16, .19, .22, .25, .23, .26, .27])

rng1 = np.random.default_rng(seed=1)
rng2 = np.random.default_rng(seed=2)
rng3 = np.random.default_rng(seed=3)
rng4 = np.random.default_rng(seed=4)

# Shuffle mean
weight_Act = rng1.choice(weight_Act, replace=False, size=12)
weight_Clr = rng1.choice(weight_Clr, replace=False, size=12)
sensitivity = rng1.choice(sensitivity, replace=False, size=12)
 
alphaAct_pos_mu = rng1.choice(learning_rate, replace=False, size=12)
alphaAct_neg_mu = rng2.choice(learning_rate, replace=False, size=12)
alphaClr_pos_mu = rng3.choice(learning_rate, replace=False, size=12)
alphaClr_neg_mu = rng4.choice(learning_rate, replace=False, size=12)

# Shuffle mean
alpha_sd = rng1.choice(alpha_sd, replace=False, size=12)
weight_sd = rng1.choice(weight_sd, replace=False, size=12)
sensitivity_sd = rng1.choice(sensitivity_sd, replace=False, size=12)

#[[[HC-Act, HC-Clr], [PD-Act, PD-Clr]]], , in shape [group, session, condition]
# [[HC, PD]], in shape [group, session]
# number of simulation
# mean
hier_weight_mu = [[weight_Act[2*(simNumber-1)], weight_Clr[2*(simNumber-1)]], [weight_Act[2*(simNumber-1)+1],weight_Clr[2*(simNumber-1)+1]]]
hier_alphaAct_pos_mu = [alphaAct_pos_mu[2*(simNumber-1)],alphaAct_pos_mu[2*(simNumber-1)+1]]
hier_alphaAct_neg_mu = [alphaAct_neg_mu[2*(simNumber-1)],alphaAct_neg_mu[2*(simNumber-1)+1]]
hier_alphaClr_pos_mu = [alphaClr_pos_mu[2*(simNumber-1)],alphaClr_pos_mu[2*(simNumber-1)+1]]
hier_alphaClr_neg_mu = [alphaClr_pos_mu[2*(simNumber-1)],alphaClr_neg_mu[2*(simNumber-1)+1]]
hier_sensitivity_mu = [sensitivity[2*(simNumber-1)], sensitivity[2*(simNumber-1)+1]]

# sd
hier_alpha_sd = [alpha_sd[2*(simNumber-1)], alpha_sd[2*(simNumber-1)+1]]
hier_weight_sd = [[weight_sd[2*(simNumber-1)], weight_sd[2*(simNumber-1)]], [weight_sd[2*(simNumber-1)+1],weight_sd[2*(simNumber-1)+1]]]
hier_sensitivity_sd = [sensitivity_sd[2*(simNumber-1)], sensitivity_sd[2*(simNumber-1)+1]]


 
"""True values of individual-level parameters are randomly taken from predefined hierarchical level parameters, 
Therfpre, call trueParamAllParts function to generate and save true parameters for each participant"""
generating_hier_grand_truth(hier_weight_mu=hier_weight_mu, hier_alphaAct_pos_mu=hier_alphaAct_pos_mu, hier_alphaAct_neg_mu=hier_alphaAct_neg_mu,
                              hier_alphaClr_pos_mu=hier_alphaClr_pos_mu, hier_alphaClr_neg_mu=hier_alphaClr_neg_mu,
                              hier_sensitivity_mu=hier_sensitivity_mu, hier_alpha_sd=hier_alpha_sd, hier_weight_sd=hier_weight_sd, hier_sensitivity_sd=hier_sensitivity_sd,
                              simNumber=simNumber)

# The Final step is to simulate data from the grand truth parameters that has been generated from previous step
simulate_hier_rl(simNumber=simNumber)
 
"""Pooling data all data and then save it"""
# read opbervation to take out the participnats
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/AllBehData/rawBehAll.csv')
# list of subjects
subList = rawBehAll['sub_ID'].unique()
 # Dataframe for concatenating data
dataAll = pd.DataFrame([])
# Loop over list of participatns
for subName in subList:
    # Directory of simulated data
    parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation'
    # Directory of the especifit simulated participant
    dirc = f'{parent_dir}/{str(simNumber)}/{subName}/{subName}-sim-task-design-true-param.csv'
    # Read the simulated participant
    data = pd.read_csv(dirc)
    # Set the new column of the participants' name
    data['sub_ID'] = subName
 
    # Concatenating each dataframe
    dataAll = pd.concat([dataAll, data])    
    
# Save concatenated data over all particiapnts
dataAll.to_csv(f'{parent_dir}/{str(simNumber)}/{str(simNumber)}-all-sim-task-design-true-param.csv', index=False)

# List of subjects
rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/AllBehData/rawBehAll.csv')
# list of subjects
subList = rawBehAll['sub_ID'].unique()
 
for subName in subList:
    # Directory of simulated data
    parent_dir  = '/mnt/scratch/projects/7TPD/amin/simulation'
    # Directory of the especifit simulated participant
    dirc = f'{parent_dir}/{str(simNumber)}/{subName}/{subName}-sim-task-design-true-param.csv'
    # Read the excel file
    data = pd.read_csv(dirc)
    # Condition sequences for each particiapnt
    blocks = data.groupby(['session', 'run'])['block'].unique().to_numpy()
    blocks = np.array([blocks[0], blocks[1], blocks[2], blocks[3]]).flatten()
    #save file name
    saveFile = f'{parent_dir}/{str(simNumber)}/{subName}/{subName}-achieva7t_task-DA_beh.png'

    # Plot by a pre implemented madule
    plots.plotChosenCorrect(data = data, blocks = blocks, subName = subName, saveFile = saveFile)