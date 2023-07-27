import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os

def softmax(values, beta = 1):
    """Choices are selected by the soft-max function rule in each trial"""
    nom = np.exp(values[0]*beta)
    denom = np.sum([nom + np.exp(values[1]*beta)])
    return nom/denom


def sample_Bernouli(theta = .5, n_samples = 1):
    """
    Generating samples at random from Bernouli density funtion
    """
    return (np.random.rand(n_samples) <= theta).astype(int) 
   
    
def trueParamAllParts(alphaAct_mu, alphaAct_sd,
                      alphaClr_mu, alphaClr_sd,
                      weghtAct_mu, weghtAct_sd,
                      beta_mu, beta_sd, simNumber = 1):
    """generate and put individual and heirarchical true parameters into task desgin for each participant"""
    
    # List of subjects
    subList = ['sub-004', 'sub-020', 'sub-012', 'sub-020', 'sub-025', 'sub-026', 'sub-029',
               'sub-030', 'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042',
               'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059',
               'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071',
               'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080',
               'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089',
               'sub-090', 'sub-092', 'sub-108', 'sub-109']
    try:
        # Set true parameters for each session and conditions realted to unkown parameters
        for subName in subList:
            # Get the partisipant's task design from the original behavioral dataset 'originalfMRIbehFiles'
            task_design = getTaskDesign(subName=subName)
            # Put true parameters into the task design
            task_desin_parameters = trueParam(task_design = task_design,
                                              alphaAct_mu = alphaAct_mu, alphaAct_sd = alphaAct_sd,
                                              alphaClr_mu = alphaClr_mu, alphaClr_sd = alphaClr_sd,
                                              weghtAct_mu = weghtAct_mu, weghtAct_sd = weghtAct_sd,
                                              beta_mu = beta_mu, beta_sd = beta_sd) 
            # Save task design plus true parameters for each participant
            parent_dir  = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/simulation/'
            # Check directory of subject name forlder and simulation number
            if not os.path.isdir(parent_dir + subName):
                os.mkdir(parent_dir + subName) 
            if not os.path.isdir(parent_dir + subName + '/' + str(simNumber)):
                os.mkdir(parent_dir + subName + '/' + str(simNumber))

            task_desin_parameters.to_csv(parent_dir + subName + '/' + str(simNumber) + '/' +subName +'-task-design-true-param.csv', index=False)

        # Save hierarchical true parameters
        dicHierMeanStdParam= ({'label':['Act-session1', 'Stim-session1', 'Act-session2', 'Stim-session2'],
                               'hierAlphaAct_mu':alphaAct_mu.flatten(),
                               'hierAlphaAct_sd': np.repeat(alphaAct_sd, 4),
                               'hierAlphaClr_mu': alphaClr_mu.flatten(),
                               'hierAlphaClr_sd': np.repeat(alphaClr_sd, 4),
                               'hierWeghtAct_mu': weghtAct_mu.flatten(),
                               'hieWeghtAct_sd': np.repeat(weghtAct_sd, 4),
                               'hierbeta_mu': np.repeat(beta_mu,2),
                               'hieBeta_sd': np.repeat(beta_sd, 4)})
        dataHierMeanStdParam = pd.DataFrame(dicHierMeanStdParam)

        if not os.path.isdir(parent_dir + 'hierParam'):
            os.mkdir(parent_dir + 'hierParam') 

        if not os.path.isdir(parent_dir + 'hierParam' + '/' + str(simNumber)):
            os.mkdir(parent_dir + 'hierParam' + '/' + str(simNumber))

        dataHierMeanStdParam.to_csv(parent_dir + 'hierParam' + '/' + str(simNumber) + '/hier-Mean-Std-True-Param.csv', index=False)
    
        return print("All true parameters for each participant have been generated successfully!")
    
    except:
        return print("An exception accured: " + str(e))
    
    
def getTaskDesign(subName = 'sub-092'):
    """Extract task design dataframe from a participant"""
                   
    # List of existing .csv files for each session and run realted to the subject
    files = ['/ses-02achieva7t/' + subName + '_ses-02achieva7t_task-DA_run-1_beh.csv',
             '/ses-02achieva7t/' + subName + '_ses-02achieva7t_task-DA_run-2_beh.csv',
             '/ses-03achieva7t/' + subName + '_ses-03achieva7t_task-DA_run-1_beh.csv',
             '/ses-03achieva7t/' + subName + '_ses-03achieva7t_task-DA_run-2_beh.csv']
    # Main directory of the subject
    subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
    # Making empty Dataframe to be concatenated in all four .csv file of the subject
    data = pd.DataFrame([])
    for i in range(len(files)):
        dirc = subMainDirec + subName + files[i]
        df = pd.read_csv(dirc)
        data = pd.concat([data, df])
    # Remove anu spance between column name              
    data.columns = data.columns.str.replace(' ', '')
    # Extract predefined task design     
    task_design = data[['session', 'run', 'stimActFirst', 'block', 'stimActBlock', 'trialNumber', 'yellowOnLeftSide', 'leftCanBePushed', 'winAmtLeft', 'winAmtRight', 'winAmtYellow', 'winAmtBlue', 'winAmtPushable', 'winAmtPullable', 'yellowCorrect', 'pushCorrect']]
                
    return task_design


def trueParam(task_design,
              alphaAct_mu, alphaAct_sd,
              alphaClr_mu, alphaClr_sd,
              weghtAct_mu, weghtAct_sd,
              beta_mu, beta_sd): 
    
    """
    Set true parameters in each condition and session independently.


    Arguments: 
        task_design: Dataframe
        alphaAct_mu: float Mean of learnig rate for action leaning rate
        alphaAct_sd: STD of learnig rate for action leaning rate
        ....
        
    ---------------------------
    output: Dataframe
    
        Dataframe containing the original task design plus four new colums related to four unkown parameters:

        True values of unkown parameters are generated from truncated normal distributions

        alphaAct ~ normal(alphaAct_mu, alphaAct_sd) T(0,1)

        alphaClr ~ normal(alphaClr_mu, alphaClr_sd) T(0,1).

        weghtAct ~ normal(weghtAct_mu, weghtAct_sd) T(0,1).

        beta ~ normal(beta_mu, beta_sd) T(0,10].
    
    """
    
    task_design[['alphaAct', 'alphaClr', 'weghtAct', 'beta']] = ""    
    
    nses = 2
    ncond = 2
    condition = ['Act', 'Stim']
    
    # Get random sample for each unkhown parameters from a truncated normal distribution
    # True parameters are generated for each conition and session independently
    for s in range(nses):
        
        while (True):
            # Sensitivity parameter chnage across session not condition
            beta = np.round(np.random.normal(beta_mu[s], beta_sd), 2)
            if beta > 0 and beta <= 3:
                break
        task_design.loc[task_design['session'] == s+1, 'beta'] = beta
            
        for c in range(ncond):
            while (True):
                alphaAct = np.round(np.random.normal(alphaAct_mu[s, c], alphaAct_sd), 2)
                if alphaAct > 0 and alphaAct < 1:
                    break
                    
            while (True):
                alphaClr = np.round(np.random.normal(alphaClr_mu[s, c], alphaClr_sd), 2)
                if alphaClr > 0 and alphaClr < 1:
                    break
            
            while (True):
                weghtAct = np.round(np.random.normal(weghtAct_mu[s, c], weghtAct_sd), 2)
                if weghtAct > 0 and weghtAct < 1:
                    break

            # Put generated true parameters within the origonal task design dataframe
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'alphaAct'] = alphaAct
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'alphaClr'] = alphaClr
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'weghtAct'] = weghtAct
                    
    return task_design


def simulateActClr(task_design_param):
    """Simulated data from the predifed true parameters in dataframe task_design_param"""

    # Number of trials
    n_trials = task_design_param.shape[0]
    
    # condition
    block = task_design_param.block.to_numpy()
    
    # which action and color are available options
    winAmtPushable = task_design_param.winAmtPushable.to_numpy()
    winAmtPullable = task_design_param.winAmtPullable.to_numpy()
    winAmtYellow = task_design_param.winAmtYellow.to_numpy()
    winAmtBlue = task_design_param.winAmtBlue.to_numpy()  
    
    # available options on left and right side
    leftCanBePushed = task_design_param.leftCanBePushed.to_numpy()
    yellowOnLeftSide = task_design_param.yellowOnLeftSide.to_numpy()
    
    # Correct responces
    pushCorrect = task_design_param.pushCorrect.to_numpy()
    yellowCorrect = task_design_param.yellowCorrect.to_numpy()
    
    # True Unkown Parameters
    alphaAct = task_design_param.alphaAct.to_numpy()
    alphaClr = task_design_param.alphaClr.to_numpy()
    weghtAct = task_design_param.weghtAct.to_numpy()
    beta = task_design_param.beta.to_numpy()
    
    # output of simulation
    correctChoice = np.zeros(n_trials).astype(int)
    pushed = np.zeros(n_trials).astype(int)
    yellowChosen = np.zeros(n_trials).astype(int)

    # Initial expected probability
    probPush = .5
    probPull = .5
    probYell = .5
    probBlue = .5
    
    for i in range(n_trials):
        # Standard Expected Value 
        expValuePush = probPush*winAmtPushable[i]
        expValuePull = probPull*winAmtPullable[i]
        expValueYell = probYell*winAmtYellow[i]
        expValueBlue = probBlue*winAmtBlue[i]

        # Relative contribution of Action Value Learning verus Color Value Learning
        expValuePushYell = weghtAct[i]*expValuePush + (1 - weghtAct[i])*expValueYell;
        expValuePushBlue = weghtAct[i]*expValuePush + (1 - weghtAct[i])*expValueBlue;
        expValuePullYell = weghtAct[i]*expValuePull + (1 - weghtAct[i])*expValueYell;
        expValuePullBlue = weghtAct[i]*expValuePull + (1 - weghtAct[i])*expValueBlue;

        # Calculating the soft-max function over weightening Action and Color conditions*/ 
        if (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 1) and (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 0):
            """pushed and yellow vs pulled and blue"""
            theta = softmax(values=[expValuePushYell, expValuePullBlue], beta=beta[i])
            
            # make a response
            y = sample_Bernouli(theta = theta) 
            
            # Response for the current trial
            if y==1:
                pushed[i] = 1
                yellowChosen[i] = 1
            else:
                pushed[i] = 0
                yellowChosen[i] = 0
                
        elif (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 0) or (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 1):
            """pushed and blue vs pulled and yellow"""
            theta = softmax(values=[expValuePushBlue, expValuePullYell], beta=beta[i]) 
           
            # make a response
            y = sample_Bernouli(theta = theta)        
            
            # Response for the current trial
            if y==1:
                pushed[i] = 1
                yellowChosen[i] = 0
            else:
                pushed[i] = 0
                yellowChosen[i] = 1
                
        # Get reward
        if block[i] == 'Act':
            reward = int(pushed[i] == pushCorrect[i])
            # Choice correct for the current trial
            correctChoice[i] =  reward
        elif block[i] == 'Stim':
            reward = int(yellowChosen[i] == yellowCorrect[i])
            # Choice correct for the current trial
            correctChoice[i] =  reward
        
        # Rl rule update
        if pushed[i] == 1:
            probPush = probPush + alphaAct[i]*(reward - probPush)
            probPull = 1 - probPush           
        else:
            probPull = probPull + alphaAct[i]*(reward - probPull)
            probPush = 1 - probPull                      
        if yellowChosen[i] == 1:
            probYell = probYell + alphaClr[i]*(reward - probYell)
            probBlue = 1 - probYell
        else:
            probBlue = probBlue + alphaClr[i]*(reward - probBlue)
            probYell = 1 - probBlue  
        
    # output results
    task_design_param['correctChoice'] = correctChoice
    task_design_param['pushed'] = pushed
    task_design_param['yellowChosen'] = yellowChosen 

    return task_design_param


def simulateActClrAllParts(simNumber = 1):
    """Simulated data for each participatn based on predefined True Parameters"""
    
    # List of subjects
    subList = ['sub-004', 'sub-012', 'sub-020', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
               'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-045',
               'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 'sub-060',
               'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 'sub-074',
               'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-081',
               'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 'sub-090',
               'sub-092', 'sub-108', 'sub-109']
    try:
        # Simulation for participant
        for subName in subList:

            parent_dir  = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/simulation/'+ subName + '/' + str(simNumber) + '/'
            # Read predefined task design with true parameters
            task_design_param = pd.read_csv(parent_dir + subName +'-task-design-true-param.csv')
            # simulate data
            simulated_data = simulateActClr(task_design_param)
            simulated_data.to_csv(parent_dir + subName +'-simulated-data-with-task-design-true-param.csv', index=False)
            
        return print("All simulations have been done successfully!")
    
    except Exception as e:
        return print("An exception accured: " + str(e))
    
    
    