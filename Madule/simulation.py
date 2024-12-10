"""all arguments have 2*2 array dimension"""
import numpy as np
import pandas as pd
import os
from scipy import stats
import sys
sys.path.append('..')
from Madule import plots

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
            for session in range(2): # session:
                for condition, block in enumerate(['Act', 'Stim']):
                    task_design.loc[(task_design['session'] == session+1)& (task_design['block'] == block), 'transfer_alphaAct_pos'] = transfer_alphaAct_pos[session, condition]
                    task_design.loc[(task_design['session'] == session+1)& (task_design['block'] == block), 'transfer_alphaAct_neg'] = transfer_alphaAct_neg[session, condition]
                    task_design.loc[(task_design['session'] == session+1)& (task_design['block'] == block), 'transfer_alphaClr_pos'] = transfer_alphaClr_pos[session, condition]
                    task_design.loc[(task_design['session'] == session+1)& (task_design['block'] == block), 'transfer_alphaClr_neg'] = transfer_alphaClr_neg[session, condition]
                    task_design.loc[(task_design['session'] == session+1)& (task_design['block'] == block), 'transfer_weight'] = transfer_weight[session, condition]
                    task_design.loc[(task_design['session'] == session+1)& (task_design['block'] == block), 'transfer_sensitivity'] = transfer_sensitivity[session, condition]
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
        dataHierMeanStdParam = pd.DataFrame(({'label':['Session 1 - Act', 'session 1- Stim', 'Session 2 - Act', 'session 2- Stim'],
                                              'hier_alphaAct_pos_mu':hier_alphaAct_pos_mu.flatten(),
                                              'hier_alphaAct_neg_mu': hier_alphaAct_neg_mu.flatten(),
                                              'hier_alphaClr_pos_mu': hier_alphaClr_pos_mu.flatten(),
                                              'hier_alphaClr_neg_mu': hier_alphaClr_neg_mu.flatten(),
                                              'hier_weight_mu': hier_weight_mu.flatten(),
                                              'hier_sensitivity_mu': hier_sensitivity_mu.flatten(),
                                              'hier_alpha_sd': np.repeat(hier_alpha_sd, 4),
                                              'hier_weight_sd': np.repeat(hier_weight_sd, 4),
                                              'hier_sensitivity_sd':np.repeat(hier_sensitivity_sd,4)}))
    
        # Save true parameters for hierarchical participant
        dataHierMeanStdParam.to_csv(f'{parent_dir}/{str(simNumber)}/hier-true-param.csv', index=False)
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

                    # Get reward based on the simulated response
                    if block[i] == 'Act':
                        correctChoice[i] = int(pushed[i] == pushCorrect[i])
                    elif block[i] == 'Stim':
                        correctChoice[i] = int(yellowChosen[i] == yellowCorrect[i])
                        
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
simNumber = 1

# Set mean and STD, [[Sess1-Act, Sess1-Clr], [Sess2-Act, Sess-Clr]]
hier_weight_mu = np.array([[.4, -.4], [.5,-.5]])
hier_alphaAct_pos_mu = np.array([[-1.5,-3], [-1,-3]])
hier_alphaAct_neg_mu =  np.array([[-1.5,-3], [-1,-3]])
hier_alphaClr_pos_mu = np.array([[-3,-1.5], [-3, -1]])
hier_alphaClr_neg_mu = np.array([[-3,-1.5], [-3,-1]])
hier_sensitivity_mu = np.array([[-5,-5], [-4.7,-4.7]])
hier_alpha_sd = np.array([.1])
hier_weight_sd = np.array([.1])
hier_sensitivity_sd = np.array([.1])


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