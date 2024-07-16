import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os

def set_true_all_parts(alpha_mu_arg, alpha_sd_arg,
                       weightAct_mu_arg, weightAct_sd_arg,
                       beta_mu_arg, beta_sd_arg, simNumber_arg = 1):
    """generate and put individual and heirarchical true parameters into task desgin for each participant"""
    
    # List of subjects
    subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
               'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
               'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
               'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
               'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
               'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
               'sub-090', 'sub-092', 'sub-108', 'sub-109']
    try:
        # read collected data across data
        rawBehAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')
        rawBehAll = rawBehAll.rename(columns={'leftCanBePushed                ': 'leftCanBePushed'})
        # Set true parameters for each session and conditions realted to unkown parameters
        for subName in subList:
            # Get the partisipant's task design from the original behavioral dataset 'originalfMRIbehFiles'
            task_design = rawBehAll[rawBehAll['sub_ID']==subName]
            # choose some relevant columns
            task_design = task_design[['session', 'run', 'stimActFirst', 'block', 'stimActBlock', 'trialNumber', 'yellowOnLeftSide', 'leftCanBePushed', 'winAmtLeft', 'winAmtRight', 'winAmtYellow', 'winAmtBlue', 'winAmtPushable', 'winAmtPullable', 'yellowCorrect', 'pushCorrect', 'reverse', 'group', 'patient']]
                
            # Put true parameters into the task design and then return it
            task_desin_parameters = set_true_part(task_design, alpha_mu_arg, alpha_sd_arg,
                                                  weightAct_mu_arg, weightAct_sd_arg,
                                                  beta_mu_arg, beta_sd_arg) 
            # Directory of simulated data
            parent_dir  = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/'
            # Check existing directory of subject name forlder and simulation number
            if not os.path.isdir(parent_dir + str(simNumber_arg)):
                os.mkdir(parent_dir + str(simNumber_arg)) 
            if not os.path.isdir(parent_dir + str(simNumber_arg) + '/' + subName):
                os.mkdir(parent_dir + str(simNumber_arg) + '/' + subName)
            # Save task design plus true parameters for each participant
            task_desin_parameters.to_csv(parent_dir + str(simNumber_arg) + '/' + subName + '/' +subName +'-task-design-true-param.csv', index=False)
        # Save Hierarchical true parameters
        dicHierMeanStdParam= ({'label':['Session 1 - Act', 'session 1- Stim', 'Session 2 - Act', 'session 2- Stim'],
                               'hierAlpha_mu':alpha_mu_arg.flatten(),
                               'hierAlpha_sd': np.repeat(alpha_sd_arg, 4),
                               'hierWeightAct_mu': weightAct_mu_arg.flatten(),
                               'hierWeightAct_sd': np.repeat(weightAct_sd_arg, 4),
                               'hierBeta_mu': beta_mu_arg.flatten(),
                               'hierBeta_sd': np.repeat(beta_sd_arg, 4)})
        dataHierMeanStdParam = pd.DataFrame(dicHierMeanStdParam)
    
        # Save true parameters for hierarchical participant
        dataHierMeanStdParam.to_csv(parent_dir + str(simNumber_arg) + '/hier-Mean-Std-True-Param.csv', index=False)
        return print("All true parameters for each participant have been generated and saved successfully!")
    except Exception as e:
        return print("An exception accured within trueParamAllParts function: " + str(e))
 
def set_true_part(task_design,
                  alpha_mu, alpha_sd,
                  weightAct_mu, weightAct_sd,
                  beta_mu, beta_sd): 
    """Set true parameters in each condition and session independently.
    Take random sample for each unkhown parameters from a truncated normal distribution
    True parameters are generated for each condition and session independently"""
    
    # Define new columns of grand truth parameters within predefined task design 
    task_design[['alpha', 'weightAct', 'beta']] = ""  
    # Set number of sessions and conditions, and name of conditions 
    nsess = 2

    for s in range(nsess):
        for c, condition in enumerate(['Act', 'Stim']):
            # Sensitivity parameter chnages across session but not condition
            while (True):
                beta = np.round(np.random.normal(beta_mu[s,c], beta_sd), 5)
                if beta >= 0 and beta <.2:
                    break
            # Learning rate parameter 
            while (True):
                alpha = np.round(np.random.normal(alpha_mu[s,c], alpha_sd), 2)
                if alpha >= 0 and alpha <= .7:
                    break        
            if condition == 'Act':
                # Relative Contribution parameter chnages across session and condition
                while (True):
                    weightAct = np.round(np.random.normal(weightAct_mu[s,c], weightAct_sd), 2)
                    if weightAct >= .6 and weightAct <= 1:
                        break
            elif condition == 'Stim':
                # Relative Contribution parameter chnages across session and condition
                while (True):
                    weightAct = np.round(np.random.normal(weightAct_mu[s,c], weightAct_sd), 2)
                    if weightAct >= 0 and weightAct <= .4:
                        break
            # Put generated true parameters of alpha and weight Act within the predefined task design dataframe
            task_design.loc[(task_design['session'] == s+1)& (task_design['block'] == condition), 'alpha'] = alpha
            task_design.loc[(task_design['session'] == s+1)& (task_design['block'] == condition), 'weightAct'] = weightAct
            task_design.loc[(task_design['session'] == s+1)& (task_design['block'] == condition), 'beta'] = beta
    return task_design  

def simulate_data_true_params(simNumber, runNumber):
    """Simulated data for each participatn based on predefined True Parameters"""
    # List of subjects
    subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
               'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
               'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
               'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
               'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
               'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
               'sub-090', 'sub-092', 'sub-108', 'sub-109']

    try:
        # Simulation for participant
        for subName in subList:
            parent_dir  = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/' + str(simNumber) + '/' + subName + '/' 
            # Read predefined task design with true parameters
            task_design_param = pd.read_csv(parent_dir + subName +'-task-design-true-param.csv')
            # simulate data
            simulated_data = simulateActClr(task_design_param)
            simulated_data.to_csv(parent_dir + subName +'-simulated-task-design-true-param'+str(runNumber)+'.csv', index=False)
            
        return print("All simulations have been done successfully!")
    
    except Exception as e:
        return print("An exception accured: " + str(e))
  
def simulateActClr(task_design_param):
    """Simulated data from the predefined true parameters in dataframe task_design_param"""

    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Stim']: # condition
                # get some relevant part data
                task_design_param_split = task_design_param[(task_design_param['block']==condition)&(task_design_param['session']==session)&(task_design_param['reverse']==reverse)]  

                # Predefined conditions for each trial
                block = task_design_param_split.block.to_numpy()
                
                # Predefined Winning amout of reward for Action and Color options
                winAmtPushable = task_design_param_split.winAmtPushable.to_numpy()
                winAmtPullable = task_design_param_split.winAmtPullable.to_numpy()
                winAmtYellow = task_design_param_split.winAmtYellow.to_numpy()
                winAmtBlue = task_design_param_split.winAmtBlue.to_numpy()  
                
                # Predefined options on left and right side
                leftCanBePushed = task_design_param_split.leftCanBePushed.to_numpy()
                yellowOnLeftSide = task_design_param_split.yellowOnLeftSide.to_numpy()
                
                # Predefined Correct responces for Action and color options
                pushCorrect = task_design_param_split.pushCorrect.to_numpy()
                yellowCorrect = task_design_param_split.yellowCorrect.to_numpy()
                
                # Predefined Ground truth Parameters
                alpha = task_design_param_split.alpha.to_numpy()
                weightAct = task_design_param_split.weightAct.to_numpy()
                beta = task_design_param_split.beta.to_numpy()
                
                # Predefined Number of trials
                n_trials = task_design_param_split.shape[0]
 
                # Output of simulation for correct choice and Action and Color chosen
                correctChoice = np.zeros(n_trials).astype(int)
                pushed = np.zeros(n_trials).astype(int)
                yellowChosen = np.zeros(n_trials).astype(int)

                # Initial reward probability
                probPush = .5
                probPull = .5
                probYell = .5
                probBlue = .5

                # Loop over trials
                for i in range(n_trials):
                    
                    # Compute the Standard Expected Value of each seperated option 
                    expValuePush = probPush*winAmtPushable[i]
                    expValuePull = probPull*winAmtPullable[i]
                    expValueYell = probYell*winAmtYellow[i]
                    expValueBlue = probBlue*winAmtBlue[i]

                    # Relative contribution of Action Value LeexpValuePusharning verus Color Value Learning by combining the expected values of option
                    expValuePushYell = weightAct[i]*expValuePush + (1 - weightAct[i])*expValueYell;
                    expValuePushBlue = weightAct[i]*expValuePush + (1 - weightAct[i])*expValueBlue;
                    expValuePullYell = weightAct[i]*expValuePull + (1 - weightAct[i])*expValueYell;
                    expValuePullBlue = weightAct[i]*expValuePull + (1 - weightAct[i])*expValueBlue;

                    # Calculating the soft-max function based on (pushed and yellow) vs (pulled and blue) 
                    if (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 1) or (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 0):
                        # Applying soft-max function 
                        nom = np.exp(beta[i]*expValuePushYell)
                        denom = nom + np.exp(beta[i]*expValuePullBlue)
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
                        nom = np.exp(beta[i]*expValuePushBlue)
                        denom = nom + np.exp(beta[i]*expValuePullYell)
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
                        probPush = probPush + alpha[i]*(correctChoice[i] - probPush)
                        probPull = 1 - probPush 
                    elif pushed[i] == 0:
                        probPull = probPull + alpha[i]*(correctChoice[i] - probPull)
                        probPush = 1 - probPull   
                                           
                    # Rl rule update Color Action Learning values for the next trial
                    if yellowChosen[i] == 1:
                        probYell = probYell + alpha[i]*(correctChoice[i] - probYell)
                        probBlue = 1 - probYell
                    elif yellowChosen[i] == 0:
                        probBlue = probBlue + alpha[i]*(correctChoice[i] - probBlue)
                        probYell = 1 - probBlue  

                # output results
                task_design_param.loc[(task_design_param['block']==condition)&(task_design_param['session']==session)&(task_design_param['reverse']==reverse), 'correctChoice'] = correctChoice  
                task_design_param.loc[(task_design_param['block']==condition)&(task_design_param['session']==session)&(task_design_param['reverse']==reverse), 'pushed'] = pushed  
                task_design_param.loc[(task_design_param['block']==condition)&(task_design_param['session']==session)&(task_design_param['reverse']==reverse), 'yellowChosen'] = yellowChosen  
 
    return task_design_param
