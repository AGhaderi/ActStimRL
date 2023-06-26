import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os

def trueParam(alphaAct_mu, alphaAct_sd,
              alphaClr_mu, alphaClr_sd,
              weghtAct_mu, weghtAct_sd,
              beta_mu, beta_sd, simNumber = 1):
    # List of subjects
    subList = ['sub-004', 'sub-020', 'sub-012', 'sub-020', 'sub-025', 'sub-026', 'sub-029',
               'sub-030', 'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042',
               'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059',
               'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071',
               'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080',
               'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089',
               'sub-090', 'sub-092', 'sub-108', 'sub-109']

    # Set true parameters for each session and conditions realted to unkown parameters
    for subName in subList:

        # Get the partisipant's task design from the original behavioral dataset 'originalfMRIbehFiles'
        task_design = getTaskDesign(subName=subName)
        # Put true parameters into the task design
        task_desin_parameters = hierTrueParam(task_design = task_design,
                                              alphaAct_mu = alphaAct_mu, alphaAct_sd = alphaAct_sd,
                                              alphaClr_mu = alphaClr_mu, alphaClr_sd = alphaClr_sd,
                                              weghtAct_mu = weghtAct_mu, weghtAct_sd = weghtAct_sd,
                                              beta_mu = beta_mu, beta_sd = beta_sd) 
        # Save task design plus true parameters for each participant
        parent_dir  = '../data/simulation/'
        if not os.path.isdir(parent_dir + subName):
            os.mkdir(parent_dir + subName) 

        if not os.path.isdir(parent_dir + subName + '/' + str(simNumber)):
            os.mkdir(parent_dir + subName + '/' + str(simNumber))
            
        task_desin_parameters.to_csv(parent_dir + subName + '/' + str(simNumber) + '/' +subName +'-task-design-true-param.csv', index=False)
        
    # Save hierarchical true parameters
    dicHierMeanStdParam= ({'label':['Act-session1', 'Stim-session1', 'Act-session2', 'Stim-session2'],
                           'hierAlphaAct_mu':alphaAct_mu.flatten(),
                           'hierAlphaAct_sd': alphaAct_sd.flatten(),
                           'hierAlphaClr_mu': alphaClr_mu.flatten(),
                           'hierAlphaClr_sd': alphaClr_sd.flatten(),
                           'hierWeghtAct_mu': weghtAct_mu.flatten(),
                           'hieWeghtAct_sd': weghtAct_sd.flatten(),
                           'hierbeta_mu': beta_mu.flatten(),
                           'hieBeta_sd': beta_sd.flatten()})
    dataHierMeanStdParam = pd.DataFrame(dicHierMeanStdParam)
    
    if not os.path.isdir(parent_dir + 'hierParam'):
        os.mkdir(parent_dir + 'hierParam') 

    if not os.path.isdir(parent_dir + 'hierParam' + '/' + str(simNumber)):
        os.mkdir(parent_dir + 'hierParam' + '/' + str(simNumber))
            
    dataHierMeanStdParam.to_csv(parent_dir + 'hierParam' + '/' + str(simNumber) + '/hier-Mean-Std-True-Param.csv', index=False)
    
def getTaskDesign(subName = 'sub-092'):
    """Extract task design dataframe from a participant"""
                   
    # List of existing .csv files for each session and run realted to the subject
    files = ['/ses-02achieva7t/' + subName + '_ses-02achieva7t_task-DA_run-1_beh.csv',
             '/ses-02achieva7t/' + subName + '_ses-02achieva7t_task-DA_run-2_beh.csv',
             '/ses-03achieva7t/' + subName + '_ses-03achieva7t_task-DA_run-1_beh.csv',
             '/ses-03achieva7t/' + subName + '_ses-03achieva7t_task-DA_run-2_beh.csv']
    # Main directory of the subject
    subMainDirec = '../data/originalfMRIbehFiles/'
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


def hierTrueParam(task_design,
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
        for c in range(ncond):
            while (True):
                alphaAct = np.round(np.random.normal(alphaAct_mu[s, c], alphaAct_sd[s, c]), 2)
                if alphaAct > 0 and alphaAct < 1:
                    break
                    
            while (True):
                alphaClr = np.round(np.random.normal(alphaClr_mu[s, c], alphaClr_sd[s, c]), 2)
                if alphaClr > 0 and alphaClr < 1:
                    break
            
            while (True):
                weghtAct = np.round(np.random.normal(weghtAct_mu[s, c], weghtAct_sd[s, c]), 2)
                if weghtAct > 0 and weghtAct < 1:
                    break
            
            while (True):
                beta = np.round(np.random.normal(beta_mu[s, c], beta_sd[s, c]), 2)
                if beta > 0 and beta <= 10:
                    break
            # Put generated true parameters within the origonal task design dataframe
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'alphaAct'] = alphaAct
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'alphaClr'] = alphaClr
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'weghtAct'] = weghtAct
            task_design.loc[(task_design['session'] == s+1) & (task_design['block'] == condition[c]), 'beta'] = beta      
                    
    return task_design